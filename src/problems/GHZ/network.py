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

def su4_reduced(circuit: tc.Circuit, qubit_1:int, qubit_2:int, theta: jax.Array) -> tc.Circuit:
    """
    Apply a reduced SU(4) gate to two qubits with parameters theta.

    Args:
        circuit (tc.Circuit): The quantum circuit to which the gate will be applied.
        qubit_1 (int): The first qubit index.
        qubit_2 (int): The second qubit index.
        theta (jax.Array): A 2D array of shape (2, 3) containing the parameters for the gate.

    Returns:
        tc.Circuit: The updated quantum circuit with the reduced SU(4) gate applied.
    """
    circuit.rx(qubit_1, theta=theta[0])
    circuit.ry(qubit_1, theta=theta[1])
    circuit.rz(qubit_1, theta=theta[2])
    circuit.rx(qubit_2, theta=theta[0])
    circuit.ry(qubit_2, theta=theta[1])
    circuit.rz(qubit_2, theta=theta[2])
    circuit.rxx(qubit_1, qubit_2, theta=theta[3])
    circuit.ryy(qubit_1, qubit_2, theta=theta[4])
    circuit.rzz(qubit_1, qubit_2, theta=theta[5])
    circuit.rx(qubit_1, theta=theta[0])
    circuit.ry(qubit_1, theta=theta[1])
    circuit.rz(qubit_1, theta=theta[2])
    circuit.rx(qubit_2, theta=theta[0])
    circuit.ry(qubit_2, theta=theta[1])
    circuit.rz(qubit_2, theta=theta[2])
    return circuit

def ZZ_measurement(circuit: tc.Circuit, qubit_1:int, qubit_2:int, theta: jax.Array):
    circuit.ry(qubit_1, theta=jnp.pi / 2)
    circuit.rxx(qubit_1, qubit_2, theta=theta[0])
    circuit.rx(qubit_1, theta=theta[1])
    circuit.rx(qubit_2, theta=theta[2])
    circuit.ryy(qubit_1, qubit_2, theta=theta[3])
    circuit.ry(qubit_1, theta=theta[4])
    circuit.ry(qubit_2, theta=theta[5])
    circuit.rzz(qubit_1, qubit_2, theta=theta[6])
    circuit.rz(qubit_1, theta=theta[7])
    circuit.rz(qubit_2, theta=theta[8])
    circuit.ry(qubit_1, theta=-jnp.pi / 2)
    circuit.barrier_instruction()
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

def unitary_vqe_circuit(params: jax.Array, n_bits: int) -> tc.Circuit:
    """
    Create a VQE circuit with a unitary block applied to the first n_bits qubits.

    Args:
        n_bits (int): The number of bits (qubits) in the circuit.
        params

    Returns:
        tc.Circuit: The quantum circuit with the unitary block applied.
    """
    p0_shape = (n_bits - 1, 5, 3)
    p0_size = math.prod(p0_shape)

    p1_shape = (n_bits, 3)

    # Slice the flat vector
    p0_flat = params[:p0_size]
    p1_flat = params[p0_size:]

    # Reshape and return
    p0 = p0_flat.reshape(p0_shape)
    p1 = p1_flat.reshape(p1_shape)

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

def unitary_vqe_circuit_depth2(params: jax.Array, n_bits: int) -> tc.Circuit:
    """
    Create a VQE circuit with a unitary block applied to the first n_bits qubits.

    Args:
        n_bits (int): The number of bits (qubits) in the circuit.
        params

    Returns:
        tc.Circuit: The quantum circuit with the unitary block applied.
    """
    p0_shape = (n_bits - 1, 5, 3)
    p0_size = math.prod(p0_shape)

    p2_shape = (n_bits, 3)

    # Slice the flat vector
    p0_flat = params[:p0_size]
    p1_flat = params[p0_size:2*p0_size]
    p2_flat = params[2*p0_size:]

    # Reshape and return
    p0 = p0_flat.reshape(p0_shape)
    p1 = p1_flat.reshape(p0_shape)
    p2 = p2_flat.reshape(p2_shape)

    circuit = tc.Circuit(n_bits)
    for i in range(n_bits):
        circuit.h(i)
    unitary_block(circuit, n_bits, p0)
    unitary_block(circuit, n_bits, p1)

    # Apply a final layer of rotations to all qubits
    for i in range(n_bits):
        circuit.rx(i, theta=p2[i, 0])
        circuit.ry(i, theta=p2[i, 1])
        circuit.rz(i, theta=p2[i, 2])

    return circuit

def init_unitary_vqe_param(n_bits:int, randkey: Any) -> jax.Array:
    p0_shape = (n_bits - 1, 5, 3)
    p0_size = jnp.prod(jnp.array(p0_shape))  # More robust way to calculate size

    p1_shape = (n_bits, 3)
    p1_size = jnp.prod(jnp.array(p1_shape))
    total_params = int(p0_size + p1_size)
    return jax.random.uniform(randkey, shape=(total_params,), minval=-jnp.pi, maxval=jnp.pi)

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

# ----------------------------------LOCC-VQE Circuit (Sampling)----------------------------------

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

def ancilla_prep_reduced(circuit:tc.Circuit, theta: jax.Array, target_1:int, target_2:int, ancilla:int) -> tc.Circuit:
    """
    Prepare an ancilla state using a reduced SU(4) gate on two target qubits and an ancilla qubit.

    Args:
        circuit (tc.Circuit): The quantum circuit to which the gate will be applied.
        theta (jax.Array): A 2D array of shape (2, 3) containing the parameters for the reduced SU(4) gates.
        target_1 (int): The first target qubit index.
        target_2 (int): The second target qubit index.
        ancilla (int): The ancilla qubit index.
    Returns:
        tc.Circuit: The updated quantum circuit with the ancilla preparation applied.
    """
    su4_reduced(circuit, target_1, ancilla, theta[0])
    su4_reduced(circuit, target_2, ancilla, theta[1])
    return circuit

def ancilla_prep_paper(circuit:tc.Circuit, theta: jax.Array, target_1:int, target_2:int, ancilla:int) -> tc.Circuit:
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
    ZZ_measurement(circuit, target_1, ancilla, theta[0])

    # ── second half (acts on data-1 & ancilla) ─────────────────────────────
    ZZ_measurement(circuit, target_2, ancilla, theta[1])

    return circuit

def syndrome_circuit_reduced(circuit:tc.Circuit, n_qubits:int, ancilla_params: jax.Array) -> tc.Circuit:
    """
    Prepare a syndrome circuit for `n_bits` data qubits, using `n_bits-1` ancilla.

    Args:
        circuit (tc.Circuit): The quantum circuit to which the syndrome preparation will be applied.
        ancilla_params (jax.Array): A 2D array of shape (n_bits-1, 2, 3) containing the parameters for the ancilla preparation.
        n_qubits (int): The number of data qubits (n_bits data + n_bits ancilla).
    Returns:
        tc.Circuit: The updated quantum circuit with the syndrome preparation applied.
    """
    ancilla_params = ancilla_params.reshape((n_qubits - 1, 2, 6))
    for i in range(n_qubits):
        circuit.h(i)

    for i in range(n_qubits - 1):
        ancilla_prep_reduced(circuit, ancilla_params[i], i, i + 1, n_qubits + i)
    return circuit

def syndrome_circuit_paper(circuit:tc.Circuit, n_qubits:int, ancilla_params: jax.Array) -> tc.Circuit:
    """
    Prepare a syndrome circuit for `n_bits` data qubits, using `n_bits-1` ancilla.

    Args:
        circuit (tc.Circuit): The quantum circuit to which the syndrome preparation will be applied.
        ancilla_params (jax.Array): A 2D array of shape (n_bits-1, 2, 3) containing the parameters for the ancilla preparation.
        n_qubits (int): The number of data qubits (n_bits data + n_bits ancilla).
    Returns:
        tc.Circuit: The updated quantum circuit with the syndrome preparation applied.
    """
    ancilla_params = ancilla_params.reshape((n_qubits - 1, 2, 9))
    for i in range(n_qubits):
        circuit.h(i)

    for i in range(n_qubits - 1):
        ancilla_prep_paper(circuit, ancilla_params[i], i, i + 1, n_qubits + i)
    return circuit

# ----------------------------------Sample Circuit----------------------------------

def new_syndrome_circuit(n_qubits: int, ancilla_params: jax.Array) -> tc.Circuit:
    """
    Prepare a syndrome circuit for `n_bits` data qubits, using `n_bits-1` ancilla.

    Args:
        n_qubits (int): The number of data qubits (n_bits data + n_bits ancilla).
        ancilla_params (jax.Array): A 2D array of shape (n_bits-1, 2, 5, 3) containing the parameters for the ancilla preparation.
    Returns:
        tc.Circuit: The quantum circuit with the syndrome preparation applied.
    """
    circuit = tc.Circuit(2 * n_qubits)
    ancilla_params = ancilla_params.reshape((n_qubits - 1, 2, 5, 3))
    p0 = ancilla_params[:, 0, :, :].reshape((n_qubits - 1, 5, 3))
    p1 = ancilla_params[:, 1, :, :].reshape((n_qubits - 1, 5, 3))
    for i in range(n_qubits):
        circuit.h(i)
    unitary_block(circuit, n_qubits, p0)
    for i in range(n_qubits - 1):
        circuit.crx(i, n_qubits + i, theta=p1[i, 0, 0])
        circuit.cry(i, n_qubits + i, theta=p1[i, 0, 1])
        circuit.crx(1 + i, i + n_qubits, theta=p1[i, 1, 0])
        circuit.cry(1 + i, i + n_qubits, theta=p1[i, 1, 1])
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
    p0 = p0.reshape((n_bits, 2, 3))
    for i in range(n_bits - 1):
        circuit.rxx(i, i+1, theta=p0[i, 0, 0])
        circuit.ryy(i, i+1, theta=p0[i, 0, 1])
        circuit.rzz(i, i+1, theta=p0[i, 0, 2])
    for i in range(n_bits):
        su2(circuit, i, p0[i, 1])
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



def correction_circuit_qsim(n, corr_circuit, params_corr_1, params_corr_2, params_corr_3):
    m = int(math.log2(n))  # number of layers in the binary tree
    for i in range(m):
        count = 0
        for j in range(1, n, 2 ** (i + 1)):
            for k in range(2 ** i):
                corr_circuit.ry(n + j, theta=jnp.pi / 2)
                corr_circuit.rxx(n + j, j + k + 2 ** i - 1, theta=params_corr_1[i][count][0])
                corr_circuit.rx(j + k + 2 ** i - 1, theta=params_corr_1[i][count][1])
                corr_circuit.ry(n + j, theta=-jnp.pi / 2)
            if j + 2 ** i < n:
                corr_circuit.cnot(n + j + 2 ** i, n + j)
            count = count + 1

    for i in range(n):
        for j in range(n):
            corr_circuit.ry(n + i, theta=jnp.pi / 2)
            corr_circuit.rxx(n + i, j, theta=params_corr_2[i][j][0])
            corr_circuit.rx(j, theta=params_corr_2[i][j][1])  # previous mistakenly writen i instead of j here!!!
            corr_circuit.ry(n + i, theta=-jnp.pi / 2)

    for i in range(n):
        corr_circuit.rx(i, theta=params_corr_3[i][0])
        corr_circuit.ry(i, theta=params_corr_3[i][1])
        corr_circuit.rz(i, theta=params_corr_3[i][2])
    return corr_circuit

def reshape_params(flat: jnp.ndarray, n: int):
    """Slice flat→(p0,p1,p2,p3) with the same shapes you used in Qiskit."""
    # shapes
    shape0 = (n - 1, 2, 6)
    shape1 = (int(math.log2(n)), n // 2, 3)
    shape2 = (n, n, 2)
    shape3 = (n, 3)
    # sizes
    s0 = math.prod(shape0)
    s1 = math.prod(shape1)
    s2 = math.prod(shape2)
    # slices
    p0 = flat[:s0].reshape(shape0)
    p1 = flat[s0 : s0 + s1].reshape(shape1)
    p2 = flat[s0 + s1 : s0 + s1 + s2].reshape(shape2)
    p3 = flat[s0 + s1 + s2 :].reshape(shape3)
    return p0, p1, p2, p3



def init_reduced_syndrome_correction_parameters(n_bits: int, randkey: Any) -> jax.Array:
    """
    Initialize the parameters for the reduced syndrome correction circuit.

    Args:
        n_bits (int): The number of bits (qubits) in the circuit.
        randkey (Any): A JAX random key for parameter initialization.

    Returns:
        jax.Array: A 1D array of shape (n_bits * 3 + (n_bits - 1) * 30) containing the initialized parameters for the correction circuit.
    """
    p0_shape = (n_bits - 1, 2, 9)
    p0_size = math.prod(p0_shape)

    p1_shape = (int(math.log2(n_bits)), n_bits // 2, 3)
    p1_size = math.prod(p1_shape)

    p2_shape = (n_bits, n_bits, 2)
    p2_size = math.prod(p2_shape)

    p3_shape = (n_bits, 3)
    p3_size = math.prod(p3_shape)

    total_params = int(p0_size + p1_size + p2_size + p3_size)
    return jax.random.uniform(randkey, shape=(total_params,), minval=-jnp.pi, maxval=jnp.pi)

def reshape_params_paper(flat: jnp.ndarray, n: int):
    """Slice flat→(p0,p1,p2,p3) with the same shapes you used in Qiskit."""
    # shapes
    shape0 = (n - 1, 2, 9)
    shape1 = (int(math.log2(n)), n // 2, 3)
    shape2 = (n, n, 2)
    shape3 = (n, 3)
    # sizes
    s0 = math.prod(shape0)
    s1 = math.prod(shape1)
    s2 = math.prod(shape2)
    # slices
    p0 = flat[:s0].reshape(shape0)
    p1 = flat[s0 : s0 + s1].reshape(shape1)
    p2 = flat[s0 + s1 : s0 + s1 + s2].reshape(shape2)
    p3 = flat[s0 + s1 + s2 :].reshape(shape3)
    return p0, p1, p2, p3

def paper_syndrome_circuit(params, n_bits):
    # Extract parameters for the syndrome circuit
    p0, p1, p2, p3 = reshape_params_paper(params, n_bits)
    # Create the circuit
    circuit = tc.Circuit(2 * n_bits)
    # Apply the reduced syndrome circuit
    syndrome_circuit_paper(circuit, n_bits, p0)
    # Apply the correction circuit
    correction_circuit_qsim(n_bits, circuit, p1, p2, p3)

    return circuit

