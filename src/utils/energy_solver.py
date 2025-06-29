from numpy.linalg import eigvalsh
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import StatevectorEstimator as Estimator
from qiskit import QuantumCircuit

def ground_truth_solver(observable: SparsePauliOp) -> float:
    """
    Solves the ground truth of a given observable using its eigenvalues.

    Args:
        observable (SparsePauliOp): The observable for which to find the ground truth.

    Returns:
        float: The minimum eigenvalue of the observable, representing the ground truth.
    """
    solution_eigenvalue = min(eigvalsh(observable.to_matrix()))
    return solution_eigenvalue

def vqe_cost(
    circuit: QuantumCircuit,
    observable: SparsePauliOp,
    params: list
) -> float:

    estimator = Estimator()
    pub = (circuit, observable, params)
    results = estimator.run([pub]).result()[0].data.evs
    return results[0]
