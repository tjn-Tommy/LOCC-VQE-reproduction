from functools import partial
import os
os.environ["JAX_TRACEBACK_FILTERING"] = "off"
import jax
import yaml
import importlib
import optax
import problems
import tensorcircuit as tc
# Our project-specific imports
from utils import *
from problems.GHZ_tc import *
from jax import numpy as jnp
tc.set_backend("jax")

def main(config_path="./configs/jax_config.yaml"):
    """
    Main function to run a VQE experiment based on a config file.
    """
    # 1. Load configuration and select device
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
    hamiltonian = partial(tc_energy, global_term = 16, perturb = 0.2)
    reduced_hamiltonian = reduced_hamiltonian_GHZ(num_qubits, 16, 0.2)
    root_key, subkey = make_batch_keys(root_key, batch_size)
    init_simple_net_vmap = jax.vmap(init_simple_net, in_axes=(0,None), out_axes=0)
    model = SimpleNet()
    unravel = get_unravel(num_qubits)
    params= init_simple_net_vmap(subkey, num_qubits)
    unravel_vmap = jax.vmap(unravel, in_axes=(0,), out_axes=0)
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
        'theta_1': synd_params,
        'gamma': params
    }
    optimizer = optax.adam(learning_rate=config['optimizer_params']['learning_rate'])
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
    def run_training_loop(opt_state, opt_params):
        def _body(carry, idx):
            opt_state, opt_params, index = carry
            index += 1
            # Run the training step
            updates, optimizer_state = train_step_partial(
                                theta_1 = opt_params['theta_1'],
                                gamma =opt_params['gamma'],
                                optimizer_state = opt_state
                                        )
            return (optimizer_state, updates, index), (updates, index)
        # Loop over the number of iterations
        (opt_state, opt_params, _), (new_param, index) = \
            jax.lax.scan(_body, (opt_state, opt_params, 0), jnp.arange(config['optimizer_params']['iterations']))
        return new_param, index
    run_training_jit = jax.jit(run_training_loop)

    new_param, index = run_training_jit(opt_state, opt_params)
    print(index)
    print(f"--- Experiment '{exp_name}' Finished ---")

if __name__ == "__main__":
    main()