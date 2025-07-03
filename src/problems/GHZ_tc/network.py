import tensorcircuit as tc
import jax
import jax.numpy as jnp
import math
from typing import Any
tc.set_backend("jax")


# ----------------------------------SU(2) and SU(4) gates----------------------------------
def su2(circuit: tc.Circuit, qubit: int, theta: jax.Array) -> tc.Circuit:
    """
    Apply a SU(2) gate to a single qubit with parameters theta.

    Args:
        circuit (tc.Circuit): The quantum circuit to which the gate will be applied.
        qubit (int): The qubit index.
        theta (jax.Array): A 1D array of shape (3,) containing the parameters for the gate.

    Returns:
        tc.Circuit: The updated quantum circuit with the SU(2) gate applied.
    """
    circuit.rx(qubit, theta=theta[0])
    circuit.ry(qubit, theta=theta[1])
    circuit.rz(qubit, theta=theta[2])
    return circuit

def su4(circuit: tc.Circuit, qubit_1:int, qubit_2:int, theta: jax.Array) -> tc.Circuit:
    """
    Apply a SU(4) gate to two qubits with parameters theta.

    Args:
        circuit (tc.Circuit): The quantum circuit to which the gate will be applied.
        qubit_1 (int): The first qubit index.
        qubit_2 (int): The second qubit index.
        theta (jax.Array): A 2D array of shape (5, 3) containing the parameters for the gate.

    Returns:
        tc.Circuit: The updated quantum circuit with the SU(4) gate applied.
    """
    circuit.rx(qubit_1, theta=theta[0, 0])
    circuit.ry(qubit_1, theta=theta[0, 1])
    circuit.rz(qubit_1, theta=theta[0, 2])
    circuit.rx(qubit_2, theta=theta[1, 0])
    circuit.ry(qubit_2, theta=theta[1, 1])
    circuit.rz(qubit_2, theta=theta[1, 2])
    circuit.rxx(qubit_1, qubit_2, theta=theta[2, 0])
    circuit.ryy(qubit_1, qubit_2, theta=theta[2, 1])
    circuit.rzz(qubit_1, qubit_2, theta=theta[2, 2])
    circuit.rx(qubit_1, theta=theta[3, 0])
    circuit.ry(qubit_1, theta=theta[3, 1])
    circuit.rz(qubit_1, theta=theta[3, 2])
    circuit.rx(qubit_2, theta=theta[4, 0])
    circuit.ry(qubit_2, theta=theta[4, 1])
    circuit.rz(qubit_2, theta=theta[4, 2])
    return circuit


# ----------------------------------Unitary VQE Circuit----------------------------------
def unitary_block(circuit: tc.Circuit, n_bits:int, theta: jax.Array) -> tc.Circuit:
    """
    Apply a unitary block to the circuit, which consists of pairs of SU(4) gates

    Args:
        circuit (tc.Circuit): The quantum circuit to which the unitary block will be applied.
        n_bits (int): The number of bits (qubits) in the circuit.
        theta (jax.Array): A 1D array of shape (n_bits,) containing the parameters for the SU(4) gates. Shape: (n_bits - 1, 5, 3)

    Returns:
        tc.Circuit: The updated quantum circuit with the unitary block applied.
    """

    for i in range(0, n_bits - 1, 2):  # even pairs
        su4(circuit, i, i + 1, theta[i])
    for i in range(1, n_bits - 1, 2):  # odd pairs
        su4(circuit, i, i + 1, theta[i])
    return circuit

def unitary_vqe_circuit(n_bits: int, p0: jax.Array, p1: jax.Array) -> tc.Circuit:
    """
    Create a VQE circuit with a unitary block applied to the first n_bits qubits.

    Args:
        n_bits (int): The number of bits (qubits) in the circuit.
        p0 (jax.Array): A 1D array of shape (n_bits - 1, 5, 3) containing the parameters for the SU(4) gates.

    Returns:
        tc.Circuit: The quantum circuit with the unitary block applied.
    """
    circuit = tc.Circuit(n_bits)
    for i in range(n_bits):
        circuit.h(i)
    unitary_block(circuit, n_bits, p0)

    # Apply a final layer of rotations to all qubits
    for i in range(n_bits):
        circuit.rx(i, theta=p1[i, 0])
        circuit.ry(i, theta=p1[i, 1])
        circuit.rz(i, theta=p1[i, 2])

    return circuit

def get_unitary_vqe_params(n_bits: int) -> int:
    """
    Calculate the total number of trainable parameters for the unitary VQE circuit.

    Args:
        n_bits (int): The number of bits (qubits) in the circuit.

    Returns:
        int: The total number of parameters.
    """
    if n_bits < 2:
        return 0
    # Each SU(4) gate has 15 parameters, and there are (n_bits - 1) such gates
    su4_params = (n_bits - 1) * 15
    # Each qubit has 3 additional rotation parameters
    rotation_params = n_bits * 3
    return su4_params + rotation_params

def long_range_block(n_bits: int, thetas: jax.Array, max_stride: int | None = None) -> tc.Circuit:
    """
    Brick-wall SU(4) layers with exponentially growing stride.

    Parameters
    ----------
    n_bits : int
        Number of qubits.
    thetas : (n_layers, n_pairs_per_layer, 15) array
        15 Euler angles for each SU(4) gate.
        *n_layers* is usually ceil(log2(n_bits)).
    max_stride : int, optional
        Highest distance to entangle (defaults to largest power of two < n_bits).

    Returns
    -------
    tc.Circuit
    """
    if max_stride is None:
        max_stride = 1 << ((n_bits-1).bit_length() - 1)

    circuit = tc.Circuit(n_bits)
    layer = 0
    stride = 1

    while stride <= max_stride:
        # even sub-lattice for this stride
        for i in range(0, n_bits - stride, 2 * stride):
            su4(circuit, i, i + stride, thetas[layer, i // (2 * stride)])
        # odd sub-lattice
        for i in range(stride, n_bits - stride, 2 * stride):
            su4(circuit, i, i + stride, thetas[layer, (i - stride) // (2 * stride)])
        layer += 1
        stride <<= 1

    return circuit


# ----------------------------------LOCC-VQE Circuit----------------------------------
def ancilla_prep(circuit:tc.Circuit, theta: jax.Array, target_1:int, target_2:int, ancilla:int) -> tc.Circuit:
    """
    Prepare an ancilla state using a SU(4) gate on two target qubits and an ancilla qubit.

    Args:
        circuit (tc.Circuit): The quantum circuit to which the gate will be applied.
        theta (jax.Array): A 2D array of shape (2, 5, 3) containing the parameters for the SU(4) gates.
        target_1 (int): The first target qubit index.
        target_2 (int): The second target qubit index.
        ancilla (int): The ancilla qubit index.
    Returns:
        tc.Circuit: The updated quantum circuit with the ancilla preparation applied.
    """

    # ── first half (acts on data-0 & ancilla) ───────────────────────────────
    su4(circuit, target_1, ancilla, theta[0])

    # ── second half (acts on data-1 & ancilla) ─────────────────────────────
    su4(circuit, target_2, ancilla, theta[1])

    return circuit

def syndrome_circuit(circuit:tc.Circuit, n_qubits:int, ancilla_params: jax.Array) -> tc.Circuit:
    """
    Prepare a syndrome circuit for `n_bits` data qubits, using `n_bits-1` ancilla.

    Args:
        circuit (tc.Circuit): The quantum circuit to which the syndrome preparation will be applied.
        ancilla_params (jax.Array): A 2D array of shape (n_bits-1, 2, 5, 3) containing the parameters for the ancilla preparation.
        n_qubits (int): The number of data qubits (n_bits data + n_bits ancilla).
    Returns:
        tc.Circuit: The updated quantum circuit with the syndrome preparation applied.
    """
    ancilla_params = ancilla_params.reshape((n_qubits - 1, 2, 5, 3))
    for i in range(n_qubits):
        circuit.h(i)

    for i in range(n_qubits - 1):
        ancilla_prep(circuit, ancilla_params[i], i, i + 1, n_qubits + i)
    return circuit

def init_syndrome_parameters(n_bits: int, randkey: Any) -> jax.Array:
    """
    Initialize the parameters for the syndrome circuit.

    Args:
        n_bits (int): The number of bits (qubits) in the circuit.
        randkey (Any): A JAX random key for parameter initialization.

    Returns:
        jax.Array: A 2D array of shape (n_bits - 1, 2, 5, 3) containing the initialized parameters for the ancilla preparation.
    """
    # Each SU(4) gate has 15 parameters, and there are (n_bits - 1) such gates
    return jax.random.uniform(randkey, shape=((n_bits - 1)*30,), minval=-jnp.pi, maxval=jnp.pi)

def post_sample_correction(circuit: tc.Circuit, n_bits: int, p0: jax.Array) -> tc.Circuit:
    """
    Apply a post-sample correction to the circuit, which consists of SU(2) gates on the first n_bits qubits.

    Args:
        circuit (tc.Circuit): The quantum circuit to which the correction will be applied.
        n_bits (int): The number of bits (qubits) in the circuit.
        p0 (jax.Array): A 2D array of shape (n_bits, 3) containing the parameters for the SU(2) gates.

    Returns:
        tc.Circuit: The updated quantum circuit with the correction applied.
    """
    p0 = p0.reshape((n_bits, 3))
    for i in range(n_bits):
        su2(circuit, i, p0[i])
    return circuit

def syndrome_circuit_wrapper(n_bits: int, p0: jax.Array) -> tc.Circuit:
    """
    Wrapper for the syndrome circuit that prepares the ancilla state and applies the post-sample correction.

    Args:
        n_bits (int): The number of bits (qubits) in the circuit.
        p0 (jax.Array): A 2D array of shape [(n_bits - 1) * 30] containing the parameters for the ancilla preparation.

    Returns:
        tc.Circuit: The quantum circuit with the syndrome preparation and post-sample correction applied.
    """
    circuit = tc.Circuit(2 * n_bits)
    p0 = p0.reshape((n_bits - 1, 2, 5, 3))
    syndrome_circuit(circuit, n_bits, p0)
    return circuit

def post_sample_correction_wrapper(n_bits: int, p0: jax.Array) -> tc.Circuit:
    """
    Wrapper for the post-sample correction that applies SU(2) gates to the first n_bits qubits.

    Args:
        n_bits (int): The number of bits (qubits) in the circuit.
        p0 (jax.Array): A 2D array of shape (n_bits, 3) containing the parameters for the SU(2) gates.

    Returns:
        tc.Circuit: The quantum circuit with the post-sample correction applied.
    """
    circuit = tc.Circuit(n_bits)
    p0 = p0.reshape((n_bits, 3))
    post_sample_correction(circuit, n_bits, p0)
    return circuit

def correction_circuit(n_bits: int, circuit: tc.Circuit, p1: jax.Array, p2: jax.Array, p3: jax.Array) -> tc.Circuit:
    """
    Three-shell recovery circuit, faithfully mirroring the TensorCircuit
    version (binary-tree fan-out  ➜  dense ancilla→data layer  ➜  per-qubit clean-up).

    Arg:
        n_bits (int): Number of data qubits (n_bits data + n_bits ancilla).
        param_1 (np.Array): Parameters for the first shell of the correction circuit. Shape: (log2(n_bits), n_bits//2, 3).
        param_2 (np.Array): Parameters for the second shell of the correction circuit. Shape: (n_bits, n_bits, 2).
        param_3 (np.Array): Parameters for the third shell of the correction circuit. Shape: (n_bits, 3).
    """

    # ---- shell 1: binary-tree fan-out over ancilla -----------------------
    m = int(math.log2(n_bits))
    for i in range(m):
        count = 0
        for j in range(0, n_bits, 2 ** (i + 1)):
            for k in range(2 ** i):
                circuit.ry(n_bits + j, jnp.pi / 2)
                circuit.rxx(n_bits + j, n_bits + j + k + 2 ** i - 1, p1[i][count][0])
                circuit.rx(n_bits + j, p1[i][count][1])
                circuit.rx(n_bits + j + k + 2 ** i - 1, p1[i][count][2])
                circuit.ry(n_bits + j, -jnp.pi / 2)
                count += 1
            # feed-forward CNOT in the fan-out tree
            if j + 2 ** i < n_bits:
                circuit.cnot(n_bits + j + 2 ** i, n_bits + j)

    # ---- shell 2: all-to-all ancilla→data layer ---------------------------
    for anc in range(n_bits):
        for data in range(n_bits):
            circuit.ry(n_bits + anc, jnp.pi / 2)
            circuit.rxx(n_bits + anc, data, p2[anc][data][0])
            circuit.rx(n_bits + anc, p2[anc][data][1])
            circuit.ry(n_bits + anc, -jnp.pi / 2)

    # ---- shell 3: per-data-qubit clean-up --------------------------------
    for q in range(n_bits):
        circuit.rx(q, p3[q][0])
        circuit.ry(q, p3[q][1])
        circuit.rz(q, p3[q][2])
    return circuit



