from .solver_base import BaseSolver
from functools import partial
import os
from src.problems.GHZ import *
import jax
import yaml
import time
import matplotlib.pyplot as plt
import datetime
import optax
import tensorcircuit as tc
import numpy as np
# Our project-specific imports
from src.utils import ground_truth_solver, make_batch_keys, to_py
from src.utils.locc_vqe_solver import *
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


class LOCCSolver(BaseSolver):
    def __init__(self, config):
        self.exp_name = config['experiment_setup']['name']
        self.batch_size = config['experiment_setup']['num_runs']
        self.root_key = jax.random.PRNGKey(config['experiment_setup']['seed'])
        self.num_qubits = config['quantum_model']['num_qubits']
        self.global_term = config['quantum_model']['hamiltonian']['global_term']
        self.quantum_net = partial(unitary_vqe_circuit, n_bits=self.num_qubits)
        self.lr_theta1 = float(config["optimizer_params"]["lr_theta1"])
        self.lr_gamma = float(config["optimizer_params"]["lr_gamma"])
        self.decay_rate = config["optimizer_params"]["decay"]
        self.theta1_sample_round = config['experiment_setup']['theta1_sample_rounds']
        self.gamma_sample_round = config['experiment_setup']['gamma_sample_rounds']
        self.sample_round=config['experiment_setup']['energy_sample_rounds']
        self.results = None
        self.timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.ctype = jnp.dtype(dtype = config['precision']['complex'])
        self.ftype = jnp.dtype(dtype = config['precision']['float'])
        self.htype = jnp.dtype(dtype = config['precision']['energy'])
        # 5. Run the Training Loop
        print(f"Starting training for {config['optimizer_params']['iterations']} iterations...")


    def train(self, iteration, perturb):
        start_time = time.time()
        print(f"--- Starting VQE Experiment: {self.exp_name}--perturb { perturb } ---")
        print("Building quantum model...")
        energy = partial(tc_energy, global_term=self.global_term, perturb = perturb, ctype=self.ctype)
        reduced_hamiltonian = reduced_hamiltonian_GHZ(self.num_qubits, self.global_term, perturb)
        print(f"Loaded Hamiltonian: '{reduced_hamiltonian.to_list()}'")

        exact_energy = ground_truth_solver(reduced_hamiltonian)
        print(f"Exact ground state energy: {exact_energy:.6f}")

        print("Initializing parameters...")
        root_key, subkey = make_batch_keys(self.root_key, self.batch_size)
        init_simple_net_vmap = jax.vmap(init_simple_net, in_axes=(0, None), out_axes=0)
        model = SimpleNet(n_bits=self.num_qubits)
        unravel = get_unravel(self.num_qubits)
        params = init_simple_net_vmap(subkey, self.num_qubits)
        synd_net = new_syndrome_circuit
        root_key, subkey = make_batch_keys(root_key, self.batch_size)
        init_syndrome_parameters_vmap = jax.vmap(init_syndrome_parameters, in_axes=(None, 0), out_axes=0)
        synd_params = init_syndrome_parameters_vmap(self.num_qubits, subkey)
        corr_net = post_sample_correction

        # 4. Prepare and Configure the Trainer
        print("Configuring the trainer...")
        opt_params = {
            'theta_1': synd_params.astype(self.ftype),
            'gamma': params.astype(self.ftype)
        }
        optimizer = make_multi_rate_tx(
            lr_theta1 = self.lr_theta1,
            lr_gamma = self.lr_gamma,
            decay_cfg= self.decay_rate,
        )

        init_vmap = jax.vmap(optimizer.init, in_axes=(0,), out_axes=0)
        opt_state = init_vmap(opt_params)
        train_step_partial = partial(train_step,
                                     model=model,
                                     unravel=unravel,
                                     n_bits=self.num_qubits,
                                     synd=synd_net,
                                     corr=corr_net,
                                     hamiltonian=energy,
                                     theta1_sample_round=self.theta1_sample_round,
                                     gamma_sample_round=self.gamma_sample_round,
                                     optimizer=optimizer,
                                     ctype=self.ctype,
                                     ftype=self.ftype,
                                     htype=self.htype,
                                     )
        print("Trainer configured successfully.")
        # 5. Run the Training Loop
        print(f"Starting training for {iteration} iterations...")
        root_key, subkey = jax.random.split(root_key, 2)

        def run_training_loop(opt_state, opt_params):
            def _body(carry, idx):
                opt_state, opt_params, index, rootkey = carry
                index += 1
                # Run the training step
                rootkey, subkey = jax.random.split(rootkey, 2)
                updates, optimizer_state, gd_var_theta1, gd_var_gamma, mean_grad_theta1, mean_grad_gamma = train_step_partial(
                    theta_1=opt_params['theta_1'],
                    gamma=opt_params['gamma'],
                    optimizer_state=opt_state,
                    rootkey=subkey,
                )
                rootkey, subkey = jax.random.split(rootkey, 2)
                def compute_energy(_):
                    return energy_estimator(model=model,
                                       unravel=unravel,
                                       n_bits=self.num_qubits,
                                       synd=synd_net,
                                       theta_1_batched=updates['theta_1'],
                                       corr=corr_net,
                                       gamma_batched=updates['gamma'],
                                       hamiltonian=energy,
                                       sample_round=self.sample_round,
                                       input_key=subkey,
                                       ctype=self.ctype,
                                       ftype=self.ftype,
                                       htype=self.htype,
                                       )
                def skip_energy(_):
                    # return placeholders of correct shape
                    return jnp.inf, jnp.inf

                should_compute = jnp.equal(jnp.mod(index, 10), 0)
                min_energy, mean_energy = jax.lax.cond(
                    should_compute,
                    compute_energy,
                    skip_energy,
                    operand=None
                )
                return (optimizer_state, updates, index, rootkey), (index, updates, min_energy, mean_energy,
                                                                    gd_var_theta1, gd_var_gamma, mean_grad_theta1,
                                                                    mean_grad_gamma)

            # Loop over the number of iterations
            (opt_state, opt_params, _, _), (index, updates, min_energy, mean_energy, gd_var_theta1, gd_var_gamma,
                                            mean_grad_theta1, mean_grad_gamma) = \
                jax.lax.scan(_body, (opt_state, opt_params, 0, subkey),
                             jnp.arange(iteration))
            return index, updates, min_energy, mean_energy, gd_var_theta1, gd_var_gamma, mean_grad_theta1, mean_grad_gamma

        run_training_jit = jax.jit(run_training_loop)

        index, updates, min_energy, mean_energy, gd_var_theta1, gd_var_gamma, mean_grad_theta1, mean_grad_gamma = run_training_jit(
            opt_state, opt_params)
        print(f"Training completed after {iteration} iterations.")
        # 6. Save the results,
        print("Saving results...")
        end_time = time.time()
        total_cost = end_time - start_time
        minimum_energy = jnp.min(min_energy)
        results = {
            'experiment_name': self.exp_name,
            'total_cost_time': total_cost,
            'num_qubits': self.num_qubits,
            'exact_energy': exact_energy,
            'final_min_energy': minimum_energy,
            'min_energy': min_energy,
            'mean_energy': mean_energy,
            'grad_theta1': mean_grad_theta1,
            'final_gd_var_theta1': gd_var_theta1,
            'grad_gamma': mean_grad_gamma,
            'final_gd_var_gamma': gd_var_gamma,
        }

        out_dir = os.path.join("results", self.exp_name, self.timestamp_str, str(perturb))
        os.makedirs(out_dir, exist_ok=True)
        results_path = os.path.join(out_dir, f"results_{self.timestamp_str}.yaml")
        with open(results_path, 'w') as f:
            yaml.dump(to_py(results), f)
        print(f"Results saved to {results_path}")
        print("Results:", results['final_min_energy'])
        print(f"Total time taken: {total_cost:.2f} seconds")

        # 7. Plot the results
        it = np.array(index)
        min_e = np.array(min_energy)
        mask = ~np.isinf(min_e)
        it_valid = it[mask]
        min_e_valid = min_e[mask]
        mean_min_energy = np.mean(min_e_valid[-10:-1])
        results['mean_min_energy'] = mean_min_energy
        var_t1 = np.array(gd_var_theta1)
        var_g = np.array(gd_var_gamma)
        exact_e = float(exact_energy)  # scalar

        # 1) Min energy vs. iteration
        plt.figure()
        plt.plot(it_valid, min_e_valid, marker='o')
        plt.xlabel('Iteration')
        plt.ylabel('Min Energy')
        plt.title(f'{self.exp_name}: Min Energy per Iteration')
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
        plt.title(f'{self.exp_name}: Gradient Variance per Iteration')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, 'grad_var_plot.png'))
        plt.show()

        # 3) Difference between exact energy and min energy
        diff_e = exact_e - min_e  # vector of differences
        diff_e_valid = diff_e[mask]
        plt.figure()
        plt.plot(it_valid, diff_e_valid, marker='s', color='C2')
        plt.xlabel('Iteration')
        plt.ylabel('Exact − Min Energy')
        plt.title(f'{self.exp_name}: Energy Gap per Iteration')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, 'energy_gap_plot.png'))
        plt.show()

        # 4) Mean gradient vs. iteration
        mean_grad_t1 = np.array(mean_grad_theta1)
        mean_grad_g = np.array(mean_grad_gamma)
        plt.figure()
        plt.plot(it, mean_grad_t1, marker='o', label='Mean Grad(θ1)')
        plt.plot(it, mean_grad_g, marker='x', label='Mean Grad(γ)')
        plt.xlabel('Iteration')
        plt.ylabel('Mean Gradient')
        plt.title(f'{self.exp_name}: Mean Gradient per Iteration')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, 'mean_grad_plot.png'))
        plt.show()
        print(f"--- Experiment '{self.exp_name}' Finished ---")
        return results


