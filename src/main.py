from functools import partial
import os
#os.environ["JAX_TRACEBACK_FILTERING"] = "off"
from jax import config
# Must happen before any JAX imports
config.update("jax_enable_x64", True)
import jax
import yaml
import time
import matplotlib.pyplot as plt
import datetime
import importlib
import optax
import problems
import tensorcircuit as tc
import numpy as np
# Our project-specific imports
from utils import *
from problems.GHZ_tc import *
from jax import numpy as jnp
tc.set_backend("jax")

def build_schedule(init_lr: float, dcfg: dict):
    """Return either a float (no decay) or an Optax schedule."""
    if dcfg["type"] == "exponential":
        return optax.exponential_decay(
            init_value       = init_lr,
            transition_steps = dcfg["decay_steps"],
            decay_rate       = dcfg["decay_rate"],
            staircase        = dcfg.get("staircase", False),
        )
    elif dcfg["type"] == "cosine":
        return optax.cosine_decay_schedule(
            init_value       = init_lr,
            decay_steps      = dcfg["decay_steps"],
        )
    else:                      # 'none'
        return init_lr

def make_multi_rate_tx(lr_theta1: float,
                       lr_gamma: float,
                       decay_cfg: dict) -> optax.GradientTransformation:
    """Adam with *independent* schedulers for θ₁ and γ leaves."""
    # build schedules
    sched_theta1 = build_schedule(lr_theta1, decay_cfg)
    sched_gamma  = build_schedule(lr_gamma,  decay_cfg)

    transforms = {
        "theta_1": optax.adam(learning_rate=sched_theta1),
        "gamma":   optax.adam(learning_rate=sched_gamma),
        "default": optax.adam(learning_rate=sched_gamma),  # safeguard
    }

    def label_fn(params):
        def _assign(path, _):
            top = path[0]
            return top if top in ("theta_1", "gamma") else "default"
        return jax.tree_util.tree_map_with_path(_assign, params)

    return optax.multi_transform(transforms, label_fn)

def main(config_path="./configs/jax_config.yaml"):
    """
    Main function to run a VQE experiment based on a config file.
    """
    # 1. Load configuration and select device
    start_time = time.time()
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # 2) build the import path
    #    e.g. "problems.GHZ_tc.network"
    # path = f"problems.{config['quantum_model']['problem']}"
    # 3) import the module
    # mod = importlib.import_module(path)
    # 4) grab the class (or function) by name
    # Cls = getattr(mod, cfg["class"])
    # 5) instantiate / call it
    # instance = Cls(...)

    exp_name = config['experiment_setup']['name']
    batch_size = config['experiment_setup']['num_runs']
    print(f"--- Starting VQE Experiment: {exp_name} ---")
    root_key = jax.random.PRNGKey(42)

    # 2. Build the Quantum Model from the config
    print("Building quantum model...")
    num_qubits = config['quantum_model']['num_qubits']
    hamiltonian = partial(tc_energy, global_term = config['quantum_model']['hamiltonian']['global_term'],
                          perturb = config['quantum_model']['hamiltonian']['perturb']
                          )
    reduced_hamiltonian = reduced_hamiltonian_GHZ(num_qubits, config['quantum_model']['hamiltonian']['global_term'], config['quantum_model']['hamiltonian']['perturb'])
    root_key, subkey = make_batch_keys(root_key, batch_size)
    init_simple_net_vmap = jax.vmap(init_simple_net, in_axes=(0,None), out_axes=0)
    model = SimpleNet()
    unravel = get_unravel(num_qubits)
    params= init_simple_net_vmap(subkey, num_qubits)
    synd_net = syndrome_circuit_wrapper
    root_key, subkey = make_batch_keys(root_key, batch_size)
    init_syndrome_parameters_vmap = jax.vmap(init_syndrome_parameters, in_axes=(None, 0), out_axes=0)
    synd_params = init_syndrome_parameters_vmap(num_qubits, subkey)
    corr_net = post_sample_correction


    print(f"Loaded Hamiltonian: '{reduced_hamiltonian.to_list()}'")
    exact_energy = ground_truth_solver(reduced_hamiltonian)
    print(f"Exact ground state energy: {exact_energy:.6f}")

    # 4. Prepare and Configure the Trainer
    print("Configuring the trainer...")
    opt_params = {
        'theta_1': synd_params.astype(jnp.float64),
        'gamma': params.astype(jnp.float64)
    }
    optimizer = make_multi_rate_tx(
        lr_theta1=config["optimizer_params"]["lr_theta1"],
        lr_gamma=config["optimizer_params"]["lr_gamma"],
        decay_cfg=config["optimizer_params"]["decay"],
    )

    init_vmap = jax.vmap(optimizer.init, in_axes=(0,), out_axes=0)
    opt_state = init_vmap(opt_params)
    train_step_partial = partial(train_step,
                                     model=model,
                                     unravel = unravel,
                                     n_bits=num_qubits,
                                     synd=synd_net,
                                     corr=corr_net,
                                     hamiltonian=hamiltonian,
                                     sample_round=config['experiment_setup']['sample_rounds'],
                                     optimizer=optimizer,
                                     )
    print("Trainer configured successfully.")
    # 5. Run the Training Loop
    print(f"Starting training for {config['optimizer_params']['iterations']} iterations...")
    root_key, subkey = jax.random.split(root_key, 2)
    def run_training_loop(opt_state, opt_params):
        def _body(carry, idx):
            opt_state, opt_params, index, rootkey = carry
            index += 1
            # Run the training step
            rootkey, subkey = jax.random.split(rootkey, 2)
            updates, optimizer_state, gd_var_theta1, gd_var_gamma, mean_grad_theta1, mean_grad_gamma = train_step_partial(
                                theta_1 = opt_params['theta_1'],
                                gamma =opt_params['gamma'],
                                optimizer_state = opt_state,
                                rootkey = subkey,
                                        )
            rootkey, subkey = jax.random.split(rootkey, 2)
            min_energy, mean_energy \
                = energy_estimator(model=model,
                            unravel = unravel,
                            n_bits=num_qubits,
                            synd=synd_net,
                            theta_1_batched=updates['theta_1'],
                            corr=corr_net,
                            gamma_batched=updates['gamma'],
                            hamiltonian=hamiltonian,
                            sample_round=config['experiment_setup']['sample_rounds'],
                            input_key = subkey,
                             )
            return (optimizer_state, updates, index, rootkey), (index, updates, min_energy, mean_energy, gd_var_theta1, gd_var_gamma, mean_grad_theta1, mean_grad_gamma)
        # Loop over the number of iterations
        (opt_state, opt_params, _, _), (index, updates, min_energy, mean_energy, gd_var_theta1, gd_var_gamma, mean_grad_theta1, mean_grad_gamma) = \
            jax.lax.scan(_body, (opt_state, opt_params, 0, subkey), jnp.arange(config['optimizer_params']['iterations']))
        return index, updates, min_energy, mean_energy, gd_var_theta1, gd_var_gamma, mean_grad_theta1, mean_grad_gamma
    run_training_jit = jax.jit(run_training_loop)

    index, updates, min_energy, mean_energy, gd_var_theta1, gd_var_gamma, mean_grad_theta1, mean_grad_gamma = run_training_jit(opt_state, opt_params)
    print(f"Training completed after {config['optimizer_params']['iterations']} iterations.")
    # 6. Save the results,
    print("Saving results...")
    end_time = time.time()
    total_cost = end_time - start_time
    timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    minimum_energy = jnp.min(min_energy)
    results = {
        'experiment_name': exp_name,
        'total_cost_time': total_cost,
        'num_qubits': num_qubits,
        'exact_energy': exact_energy,
        'final_min_energy': minimum_energy,
        'min_energy': min_energy,
        'mean_energy': mean_energy,
        'grad_theta1': mean_grad_theta1,
        'final_gd_var_theta1': gd_var_theta1,
        'grad_gamma': mean_grad_gamma,
        'final_gd_var_gamma': gd_var_gamma,
        # 'params': updates
    }


    def to_py(obj):
        if isinstance(obj, np.ndarray) or hasattr(obj, "tolist"):
            return obj.tolist()
        if hasattr(obj, "__float__"):
            return float(obj)
        if hasattr(obj, "__int__"):
            return int(obj)
        if isinstance(obj, dict):
            return {k: to_py(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [to_py(v) for v in obj]
        return obj
        # Save the results to a YAML file
    out_dir = os.path.join("results", exp_name, timestamp_str)
    os.makedirs(out_dir, exist_ok=True)
    results_path = os.path.join(out_dir, f"results_{timestamp_str}.yaml")
    with open(results_path, 'w') as f:
        yaml.dump(to_py(results), f)
    print(f"Results saved to {results_path}")
    print("Results:", results['final_min_energy'])
    print(f"Total time taken: {total_cost:.2f} seconds")

    # 7. Plot the results
    it = np.array(index)
    min_e = np.array(min_energy)
    var_t1 = np.array(gd_var_theta1)
    var_g = np.array(gd_var_gamma)
    exact_e = float(exact_energy)  # scalar

    # 1) Min energy vs. iteration
    plt.figure()
    plt.plot(it, min_e, marker='o')
    plt.xlabel('Iteration')
    plt.ylabel('Min Energy')
    plt.title(f'{exp_name}: Min Energy per Iteration')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'min_energy_plot.png'))
    plt.show()

    # 2) Gradient variances vs. iteration
    plt.figure()
    plt.plot(it, var_t1, marker='o', label='Var(θ1)')
    plt.plot(it, var_g, marker='x', label='Var(γ)')
    plt.xlabel('Iteration')
    plt.ylabel('Gradient Variance')
    plt.title(f'{exp_name}: Gradient Variance per Iteration')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'grad_var_plot.png'))
    plt.show()

    # 3) Difference between exact energy and min energy
    diff_e = exact_e - min_e  # vector of differences

    plt.figure()
    plt.plot(it, diff_e, marker='s', color='C2')
    plt.xlabel('Iteration')
    plt.ylabel('Exact − Min Energy')
    plt.title(f'{exp_name}: Energy Gap per Iteration')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'energy_gap_plot.png'))
    plt.show()
    print(f"--- Experiment '{exp_name}' Finished ---")

    # 4) Mean gradient vs. iteration
    mean_grad_t1 = np.array(mean_grad_theta1)
    mean_grad_g = np.array(mean_grad_gamma)
    plt.figure()
    plt.plot(it, mean_grad_t1, marker='o', label='Mean Grad(θ1)')
    plt.plot(it, mean_grad_g, marker='x', label='Mean Grad(γ)')
    plt.xlabel('Iteration')
    plt.ylabel('Mean Gradient')
    plt.title(f'{exp_name}: Mean Gradient per Iteration')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'mean_grad_plot.png'))
    plt.show()
if __name__ == "__main__":
    main()