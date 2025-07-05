from functools import partial
import os
#os.environ["JAX_TRACEBACK_FILTERING"] = "off"

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
from utils import ground_truth_solver, make_batch_keys
from utils.unitary_vqe import *
from problems.GHZ import *
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

def make_multi_rate_tx(lr_params: float,
                       decay_cfg: dict) -> optax.GradientTransformation:
    """Adam with *independent* schedulers for θ₁ and γ leaves."""
    # build schedules
    sched_params = build_schedule(lr_params, decay_cfg)

    transforms = {
        "params": optax.adam(learning_rate=sched_params),
        "default": optax.adam(learning_rate=sched_params),  # safeguard
    }

    def label_fn(params):
        def _assign(path, _):
            top = path[0]
            return top if top in ("params",) else "default"
        return jax.tree_util.tree_map_with_path(_assign, params)

    return optax.multi_transform(transforms, label_fn)

def main(config_path="./configs/uvqe_config.yaml"):
    """
    Main function to run a VQE experiment based on a config file.
    """
    # 1. Load configuration and select device
    start_time = time.time()
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

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
    quantum_net = partial(unitary_vqe_circuit,n_bits = num_qubits)
    root_key, subkey = make_batch_keys(root_key, batch_size)
    init_parameters_vmap = jax.vmap(init_unitary_vqe_param, in_axes=(None, 0), out_axes=0)
    params = init_parameters_vmap(num_qubits, subkey)

    print(f"Loaded Hamiltonian: '{reduced_hamiltonian.to_list()}'")
    exact_energy = ground_truth_solver(reduced_hamiltonian)
    print(f"Exact ground state energy: {exact_energy:.6f}")

    # 4. Prepare and Configure the Trainer
    print("Configuring the trainer...")
    opt_params = {
        'params': params,
    }

    optimizer = make_multi_rate_tx(
        lr_params=config["optimizer_params"]["learning_rate"],
        decay_cfg=config["optimizer_params"]["decay"],
    )

    init_vmap = jax.vmap(optimizer.init, in_axes=(0,), out_axes=0)
    opt_state = init_vmap(opt_params)
    train_step_partial = partial(train_step,
                                     n_bits=num_qubits,
                                     circ=quantum_net,
                                     hamiltonian=hamiltonian,
                                     optimizer=optimizer,
                                     )
    print("Trainer configured successfully.")
    # 5. Run the Training Loop
    print(f"Starting training for {config['optimizer_params']['iterations']} iterations...")
    def run_training_loop(opt_state, opt_params):
        def _body(carry, idx):
            opt_state, opt_params, index= carry
            index += 1
            # Run the training step
            updates, optimizer_state, mean_grad_params = train_step_partial(
                                params = opt_params['params'],
                                optimizer_state = opt_state,
                                        )
            min_energy, mean_energy \
                = energy_estimator(
                            n_bits=num_qubits,
                            circ= quantum_net,
                            params=opt_params['params'],
                            hamiltonian=hamiltonian,
                             )
            return (optimizer_state, updates, index), (index, updates, min_energy, mean_energy, mean_grad_params)
        # Loop over the number of iterations
        (opt_state, opt_params, _), (index, updates, min_energy, mean_energy, mean_grad_params) = \
            jax.lax.scan(_body, (opt_state, opt_params, 0), jnp.arange(config['optimizer_params']['iterations']))
        return index, updates, min_energy, mean_energy, mean_grad_params
    run_training_jit = jax.jit(run_training_loop)

    index, updates, min_energy, mean_energy, mean_grad_params = run_training_jit(opt_state, opt_params)
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
        'mean_grad': mean_grad_params,
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

    # 2) Difference between exact energy and min energy
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

    # 3) Mean gradient vs. iteration
    mean_grad_t1 = np.array(mean_grad_params)
    plt.figure()
    plt.plot(it, mean_grad_t1, marker='o', label='Mean Grad')
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