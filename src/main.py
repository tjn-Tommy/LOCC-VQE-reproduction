import yaml
import numpy as np
import torch
from qiskit.circuit import ParameterVector
from qiskit_machine_learning.gradients import ParamShiftEstimatorGradient
# Qiskit imports
from qiskit_aer import AerSimulator
from qiskit.primitives import BackendEstimatorV2
from problems import *

# Our project-specific imports
from utils import *



def main(config_path="./configs/config.yaml"):
    """
    Main function to run a VQE experiment based on a config file.
    """
    # 1. Load configuration and select device
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    exp_name = config['experiment_setup']['name']
    print(f"--- Starting VQE Experiment: {exp_name} ---")

    use_cuda = config['backend_options']['device'] == 'cuda' and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Selected execution device: {device}")

    # 2. Build the Quantum Model from the config
    print("Building quantum model...")
    num_qubits = config['quantum_model']['num_qubits']
    num_ansatz_params = get_num_ghz_params(num_qubits)
    params_vec = ParameterVector('θ', length=num_ansatz_params)
    circuit = GHZ_circuit(num_qubits, params_vec)
    hamiltonian = hamiltonian_GHZ(num_qubits, 16, 0.2)
    reduced_hamiltonian = reduced_hamiltonian_GHZ(num_qubits, 16, 0.2)
    print(f"Loaded Ansatz with {num_ansatz_params} parameters.")
    print(f"Loaded Hamiltonian: '{hamiltonian.to_list()}'")

    exact_energy = ground_truth_solver(reduced_hamiltonian)
    print(f"Exact ground state energy: {exact_energy:.6f}")

    # 3. Configure Backend and Bridge to PyTorch
    print("Configuring Qiskit backend...")
    aer_sim = AerSimulator(method='statevector',
                           device='GPU' if use_cuda else 'CPU',
                           cuStateVec_enable=True,
                           blocking_enable=True,
                           )
    estimator = BackendEstimatorV2(backend=aer_sim,)
    qnn = qnn_wrapper(circuit, [hamiltonian], estimator).to(device)
    init_theta = torch.rand(num_ansatz_params, device=device) * (2 * np.pi)
    qnn.weight = torch.nn.Parameter(init_theta, requires_grad=True)
    # Wrapper function for the cost
    grad_estimator = ParamShiftEstimatorGradient(estimator)

    class EstimatorLayer(torch.autograd.Function):
        """
        A differentiable wrapper around Qiskit EstimatorV2 + ParamShiftEstimatorGradient.
        Forward:  expectation values  ⟨ψ(θ)| H |ψ(θ)⟩
        Backward: analytic parameter-shift gradient  d/dθ ⟨ψ(θ)| H |ψ(θ)⟩
        """

        @staticmethod
        def forward(ctx, params_batch: torch.Tensor) -> torch.Tensor:
            # 1.  Launch the batched Estimator job ---------------------------

            theta_list = params_batch.detach().cpu().numpy().tolist()
            pubs = [(circuit, hamiltonian, p.detach().cpu().numpy()) for p in params_batch]
            ev_job = estimator.run(pubs=pubs)

            evs_np   = [r.data.evs for r in ev_job.result()]   # shape (B,)
            vals = [a.item() for a in evs_np]
            # 2.  Save inputs for the backward pass --------------------------
            ctx.save_for_backward(params_batch)  # we need the *values*
            ctx.theta_list = theta_list  # gradient interface is NumPy

            return torch.tensor(vals,
                                device=params_batch.device,
                                dtype=params_batch.dtype)

        @staticmethod
        def backward(ctx, grad_output: torch.Tensor):
            (params_batch,) = ctx.saved_tensors

            # 1.  Compute analytic gradients via parameter-shift ------------
            circuits = [circuit] * len(ctx.theta_list)
            observables = [hamiltonian] * len(ctx.theta_list)
            grad_job = grad_estimator.run(circuits, observables, ctx.theta_list)
            grads_np = grad_job.result().gradients  # (B, n_params)
            stacked_grads_np = np.array(grads_np)
            grads = torch.tensor(stacked_grads_np,
                                 device=params_batch.device,
                                 dtype=params_batch.dtype)

            # 2.  Chain rule: dL/dθ = dL/dE · dE/dθ -------------------------
            return grad_output.unsqueeze(1) * grads  # same shape as params_batch

    def cost_function(params_batch: torch.Tensor) -> torch.Tensor:
        return EstimatorLayer.apply(params_batch)


    # 4. Prepare and Configure the Trainer
    print("Configuring the trainer...")
    num_runs = config['experiment_setup']['num_runs']
    initial_params_batch = np.random.uniform(0, 2 * np.pi, size=(num_runs, num_ansatz_params))

    batched_vqe_solver = BatchedVQE(
        cost_function=cost_function,
        initial_params_batch=initial_params_batch,
        optimizer_name=config['optimizer_params']['name'],
        learning_rate=config['optimizer_params']['learning_rate'],
        device=device
    )

    # 5. Run the Optimization and Analysis
    iterations = config['optimizer_params']['iterations']
    batched_vqe_solver.optimize(iterations=iterations, verbose=True)

    batched_vqe_solver.analyze_results(exact_energy=exact_energy)

    print(f"--- Experiment '{exp_name}' Finished ---")

if __name__ == "__main__":
    main()