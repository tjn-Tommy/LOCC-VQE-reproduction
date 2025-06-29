import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import StatevectorEstimator as Estimator

def hamiltonian_GHZ(n: int, global_coeff: float, perturb: float = 0.0) -> SparsePauliOp:
    """
    Constructs the Hamiltonian as a Qiskit SparsePauliOp object.

    Args:
        n: The number of qubits.
        global_coeff: The coefficient for the global X...X term. Avoid degeneracy by ensuring this is non-zero.
        perturb: The coefficient for the Z terms.

    Returns:
        A SparsePauliOp representing the Hamiltonian.
    """
    pauli_list = []
    coeff_list = []

    # 1. ZZ terms: sum over all adjacent pairs
    # Coefficient is -(1-h)/n
    coeff_zz = -(1 - perturb) / n
    for i in range(n - 1):
        zz_pauli = ['I'] * 2 * n
        zz_pauli[n - 1 - i] = 'Z'
        zz_pauli[n - 1 - (i + 1)] = 'Z'
        pauli_list.append("".join(zz_pauli))
        coeff_list.append(coeff_zz)

    # 2. Global X...X term
    # Coefficient is -(kx-h)/n
    coeff_x = -(global_coeff - perturb) / n
    pauli_list.append('X' * n + 'I' * n)  # Global X term on first n qubits
    coeff_list.append(coeff_x)

    # 3. Single Z terms
    # Coefficient is -h/n
    coeff_z = -perturb / n
    for i in range(n):
        z_pauli = ['I'] * n + ['I'] * n  # Z terms on first n qubits
        z_pauli[n - 1 - i] = 'Z'
        pauli_list.append("".join(z_pauli))
        coeff_list.append(coeff_z)

    return SparsePauliOp(pauli_list, coeffs=coeff_list)



def calculate_hamiltonian_expectation(
        circuit: QuantumCircuit,
        hamiltonian: SparsePauliOp
) -> float:
    """
    Calculates the expectation value of a Hamiltonian for a given circuit.

    Args:
        circuit: The quantum circuit preparing the state.
        hamiltonian: The SparsePauliOp object representing the Hamiltonian.

    Returns:
        The calculated expectation value (energy).
    """
    # Use the Estimator primitive to calculate the expectation value
    estimator = Estimator()
    job = estimator.run(pubs=[(circuit,hamiltonian)])

    # Get the result from the job
    result = job.result()

    # The result contains a list of expectation values. We only have one.
    energy = result[0].data.evs

    return energy


# --- Example Usage ---
if __name__ == '__main__':
    # Define parameters for the system
    n = 4
    kx = 1.5
    h = 0.5

    # 1. Create the Hamiltonian operator for our system
    print(f"Constructing Hamiltonian for n={n}, kx={kx}, h={h}...")
    hamiltonian_op = hamiltonian_GHZ(n=n, global_coeff=kx, perturb=h)
    # You can print the operator to see its structure
    # print("\nHamiltonian Operator:")
    # print(hamiltonian_op.to_list())

    # 2. Create a quantum circuit to test with.
    # For this example, we'll just create a simple superposition state.
    # In a real VQE, this would be your parameterized ansatz circuit.
    qc = QuantumCircuit(n)
    qc.h(range(n))
    qc.barrier()
    qc.rz(np.pi / 4, 0)
    qc.cx(0, 1)
    qc.cx(1, 2)
    qc.cx(2, 3)

    print("\nTest Circuit:")
    print(qc)

    # 3. Calculate the expectation value
    print("\nCalculating expectation value...")
    expected_energy = calculate_hamiltonian_expectation(qc, hamiltonian_op)

    print(f"\nCalculated Energy: {expected_energy}")