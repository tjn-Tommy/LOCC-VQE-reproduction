from qiskit.quantum_info import SparsePauliOp
import jax
import jax.numpy as jnp
import tensorcircuit as tc
tc.set_backend("jax")
def hamiltonian_GHZ(n: int, global_term: float, perturb: float = 0.0) -> SparsePauliOp:
    """
    Constructs the Hamiltonian as a Qiskit SparsePauliOp object.

    Args:
        n: The number of qubits.
        global_term: The coefficient for the global X...X term. Avoid degeneracy by ensuring this is non-zero.
        perturb: The coefficient for the Z terms.

    Returns:
        A SparsePauliOp representing the Hamiltonian.
    """
    pauli_list = []
    coeff_list = []

    # 1. ZZ terms: sum over all adjacent pairs
    # Coefficient is -(1-h)/n_bits
    coeff_zz = -(1 - perturb) / n
    for i in range(n - 1):
        zz_pauli = ['I'] * 2 * n
        zz_pauli[n - 1 - i] = 'Z'
        zz_pauli[n - 1 - (i + 1)] = 'Z'
        pauli_list.append("".join(zz_pauli))
        coeff_list.append(coeff_zz)

    # 2. Global X...X term
    # Coefficient is -(kx-h)/n_bits
    coeff_x = -(global_term - perturb) / n
    pauli_list.append('X' * n + 'I' * n)  # Global X term on first n_bits qubits
    coeff_list.append(coeff_x)

    # 3. Single Z terms
    # Coefficient is -h/n_bits
    coeff_z = -perturb / n
    for i in range(n):
        z_pauli = ['I'] * n + ['I'] * n  # Z terms on first n_bits qubits
        z_pauli[n - 1 - i] = 'X'
        pauli_list.append("".join(z_pauli))
        coeff_list.append(coeff_z)

    return SparsePauliOp(pauli_list, coeffs=coeff_list)

def tc_energy(circuit: tc.Circuit, n_bits:int, global_term: float, perturb: float = 0.0, ctype: jnp.dtype = jnp.complex64):
    e = 0.0
    coeff_zz = jnp.astype(- (1 - perturb) / n_bits, ctype)
    coeff_x = jnp.astype(- (global_term - perturb) / n_bits, ctype)
    coeff_z = jnp.astype(- perturb / n_bits, ctype)
    for i in range(n_bits - 1):
        e += coeff_zz * circuit.expectation_ps(z=[i, i + 1])  # <Z_iZ_{i+1}>
    for i in range(n_bits):  # OBC
        e += coeff_z * circuit.expectation_ps(x=[i])  # <Z_i>
    e += coeff_x * circuit.expectation_ps(x=list(range(n_bits)))  # <X_1 X_2 ... X_n>
    return jnp.real(e)



def reduced_hamiltonian_GHZ(n: int, global_term: float, perturb: float = 0.0) -> SparsePauliOp:
    """
    Constructs the Hamiltonian as a Qiskit SparsePauliOp object.

    Args:
        n: The number of qubits.
        global_term: The coefficient for the global X...X term. Avoid degeneracy by ensuring this is non-zero.
        perturb: The coefficient for the Z terms.

    Returns:
        A SparsePauliOp representing the Hamiltonian.
    """
    pauli_list = []
    coeff_list = []

    # 1. ZZ terms: sum over all adjacent pairs
    # Coefficient is -(1-h)/n_bits
    coeff_zz = -(1 - perturb) / n
    for i in range(n - 1):
        zz_pauli = ['I'] * n
        zz_pauli[n - 1 - i] = 'Z'
        zz_pauli[n - 1 - (i + 1)] = 'Z'
        pauli_list.append("".join(zz_pauli))
        coeff_list.append(coeff_zz)

    # 2. Global X...X term
    # Coefficient is -(kx-h)/n_bits
    coeff_x = -(global_term - perturb) / n
    pauli_list.append('X' * n)
    coeff_list.append(coeff_x)

    # 3. Single Z terms
    # Coefficient is -h/n_bits
    coeff_z = -perturb / n
    for i in range(n):
        z_pauli = ['I'] * n
        z_pauli[n - 1 - i] = 'X'
        pauli_list.append("".join(z_pauli))
        coeff_list.append(coeff_z)

    return SparsePauliOp(pauli_list, coeffs=coeff_list)
