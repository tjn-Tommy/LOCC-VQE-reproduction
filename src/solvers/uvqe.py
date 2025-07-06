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
from src.utils.unitary_vqe import *
from jax import numpy as jnp
tc.set_backend("jax")


def build_schedule(init_lr: float, dcfg: dict):
    """Return either a float (no decay) or an Optax schedule."""
    if dcfg["type"] == "exponential":
        return optax.exponential_decay(
            init_value=init_lr,
            transition_steps=dcfg["decay_steps"],
            decay_rate=dcfg["decay_rate"],
            staircase=dcfg.get("staircase", False),
        )
    elif dcfg["type"] == "cosine":
        return optax.cosine_decay_schedule(
            init_value=init_lr,
            decay_steps=dcfg["decay_steps"],
        )
    else:  # 'none'
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

class UVQESolver(BaseSolver):
    def __init__(self, config):
        self.exp_name = config['experiment_setup']['name']
        self.batch_size = config['experiment_setup']['num_runs']
        self.root_key = jax.random.PRNGKey(config['experiment_setup']['seed'])
        self.num_qubits = config['quantum_model']['num_qubits']
        self.global_term = config['quantum_model']['hamiltonian']['global_term']
        self.quantum_net = partial(unitary_vqe_circuit, n_bits=self.num_qubits)
        self.lr = config["optimizer_params"]["learning_rate"]
        self.decay_rate = config["optimizer_params"]["decay"]
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
        init_parameters_vmap = jax.vmap(init_unitary_vqe_param, in_axes=(None, 0), out_axes=0)
        params = init_parameters_vmap(self.num_qubits, subkey)

        print("Configuring the trainer...")
        opt_params = {
            'params': params,
        }

        optimizer = make_multi_rate_tx(lr_params= self.lr, decay_cfg= self.decay_rate)

        init_vmap = jax.vmap(optimizer.init, in_axes=(0,), out_axes=0)
        opt_state = init_vmap(opt_params)
        train_step_partial = partial(train_step,
                                     n_bits=self.num_qubits,
                                     circ=self.quantum_net,
                                     hamiltonian=energy,
                                     optimizer=optimizer,
                                     ctype = self.ctype,
                                     htype = self.htype,
                                     ftype = self.ftype,
                                     )
        print("Trainer configured successfully.")
        print(f"Starting training for {iteration} iterations...")

        def run_training_loop(opt_state, opt_params):
            def _body(carry, idx):
                opt_state, opt_params, index = carry
                index += 1
                # Run the training step
                updates, optimizer_state, mean_grad_params = train_step_partial(
                    params=opt_params['params'],
                    optimizer_state=opt_state,
                )

                def compute_energy(_):
                    return energy_estimator(
                        n_bits=self.num_qubits,
                        circ=self.quantum_net,
                        params=opt_params['params'],
                        hamiltonian=energy,
                        ctype=self.ctype,
                        htype=self.htype,
                        ftype=self.ftype,
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
                return (optimizer_state, updates, index), (index, updates, min_energy, mean_energy, mean_grad_params)

            # Loop over the number of iterations
            (opt_state, opt_params, _), (index, updates, min_energy, mean_energy, mean_grad_params) = \
                jax.lax.scan(_body, (opt_state, opt_params, 0), jnp.arange(iteration))
            return index, updates, min_energy, mean_energy, mean_grad_params

        run_training_jit = jax.jit(run_training_loop)

        index, updates, min_energy, mean_energy, mean_grad_params = run_training_jit(opt_state, opt_params)
        print(f"Training completed after {iteration} iterations.")
        # 6. Save the results,
        print("Saving results...")
        end_time = time.time()
        total_cost = end_time - start_time
        minimum_energy = jnp.min(mean_energy)
        results = {
            'experiment_name': self.exp_name,
            'total_cost_time': total_cost,
            'num_qubits': self.num_qubits,
            'exact_energy': exact_energy,
            'final_min_energy': minimum_energy,
            'min_energy': min_energy,
            'mean_energy': mean_energy,
            'mean_grad': mean_grad_params,
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
        exact_e = float(exact_energy)  # scalar
        mask = ~np.isinf(min_e)
        it_valid = it[mask]
        min_e_valid = min_e[mask]
        mean_min_energy = np.mean(min_e_valid[-10:-1])
        results['mean_min_energy'] = mean_min_energy
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

        # 2) Difference between exact energy and min energy
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


        # 3) Mean gradient vs. iteration
        mean_grad_t1 = np.array(mean_grad_params)
        plt.figure()
        plt.plot(it, mean_grad_t1, marker='o', label='Mean Grad')
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

