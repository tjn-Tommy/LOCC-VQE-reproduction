import yaml
import numpy as np
import torch
from qiskit.circuit import ParameterVector

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
    hamiltonian = hamiltonian_GHZ(num_qubits, 0.2, 0.1)
    print(f"Loaded Ansatz with {num_ansatz_params} parameters.")
    print(f"Loaded Hamiltonian: '{hamiltonian.to_list()}'")

    # 3. Configure Backend and Bridge to PyTorch
    print("Configuring Qiskit backend...")
    aer_sim = AerSimulator(method='statevector', device='GPU' if use_cuda else 'CPU')
    estimator = BackendEstimatorV2(backend=aer_sim)
    qnn = qnn_wrapper(circuit, [hamiltonian], estimator).to(device)

    # Wrapper function for the cost
    def cost_function(params_batch):
        """
        Computes the cost for a batch of parameters. It manually updates the QNN's
        internal weights for each instance in the batch and then calls the
        forward pass without any input data.
        """
        cost_list = []
        for params in params_batch:
            # 1. Update the internal weights of the QNN object.
            #    qnn.weight is the nn.Parameter tensor managed by TorchConnector.
            #    We update its content with the parameters from our optimizer.
            qnn.weight.data = params.to(device, dtype=qnn.weight.dtype)

            # 2. Call the forward pass with no input data, which is correct for VQE.
            #    The QNN will use its now-updated internal weights to compute the energy.
            cost = qnn()
            cost_list.append(cost)

        # 3. Stack the individual costs back into a single tensor for the optimizer.
        return torch.cat(cost_list).reshape(-1)

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
    exact_energy = config['analysis_params'].get('exact_energy')
    batched_vqe_solver.analyze_results(exact_energy=exact_energy)

    print(f"--- Experiment '{exp_name}' Finished ---")


if __name__ == "__main__":
    main()