from qiskit import QuantumCircuit
from qiskit.quantum_info import (
    Statevector,
    DensityMatrix,
    partial_trace,
    entropy,
    mutual_information
)

def _to_density_matrix(state):
    """Convert QC, Statevector, or DensityMatrix → DensityMatrix."""
    if isinstance(state, DensityMatrix):
        return state
    if isinstance(state, Statevector):
        return DensityMatrix(state)
    if isinstance(state, QuantumCircuit):
        statevector = Statevector.from_instruction(state)
        return DensityMatrix(statevector)
    raise TypeError("Expected QuantumCircuit, Statevector, or DensityMatrix")

def compute_entanglement_entropy(state, keep_qubits, base=2):
    """
    Compute entanglement entropy given a state and a list of qubits.

    Args:
        state: QuantumCircuit, Statevector, or DensityMatrix
        keep_qubits: list of subsystem indices to keep (others are traced out)
        base: log base (2 for bits, e for nats)

    Returns:
        float: von Neumann entropy of the reduced state on keep_qubits.
    """
    rho = _to_density_matrix(state)
    n = len(rho.dims())
    # trace out all qubits not in keep_qubits
    trace_out = [i for i in range(n) if i not in keep_qubits]
    rho_sub = partial_trace(rho, trace_out)
    return entropy(rho_sub, base=base)

def compute_qmi(state, subsys_a, subsys_b, base=2):
    """
    Compute I(A:B) = S(ρ_A) + S(ρ_B) – S(ρ_{AB}).

    Args:
        state: QuantumCircuit, Statevector, or DensityMatrix
        subsys_a: list of indices for subsystem A
        subsys_b: list of indices for subsystem B
        base: log base (2 for bits, e for nats)

    Returns:
        float: quantum mutual information I(A:B).
    """
    rho = _to_density_matrix(state)
    n = len(rho.dims())
    # keep only A∪B, trace out the rest
    keep = subsys_a + subsys_b
    trace_out = [i for i in range(n) if i not in keep]
    rho_ab = partial_trace(rho, trace_out)
    return mutual_information(rho_ab, base=base)
