from qiskit import QuantumCircuit
import numpy as np
import matplotlib.pyplot as plt
from qiskit.circuit import ParameterVector


def ancilla_prep(theta_1: np.ndarray, theta_2: np.ndarray) -> QuantumCircuit:
    """
    Unitary gadget that *behaves like* a ZZ-parity measurement on two data
    qubits using one ancilla.  Both halves are fully parameterised (9 angles
    each) so the sub-circuit can learn the optimal mapping during VQE.
    """
    qc = QuantumCircuit(3, name=f"ancilla_prep")

    # ── first half (acts on data-0 & ancilla) ───────────────────────────────
    qc.rx(theta_1[0], 0)
    qc.ry(theta_1[1], 0)
    qc.rz(theta_1[2], 0)
    qc.rx(theta_1[0], 2)
    qc.ry(theta_1[1], 2)
    qc.rz(theta_1[2], 2)
    qc.rxx(theta_1[3], 0, 2)
    qc.ryy(theta_1[4], 0, 2)
    qc.rzz(theta_1[5], 0, 2)
    qc.rx(theta_1[0], 0)
    qc.ry(theta_1[1], 0)
    qc.rz(theta_1[2], 0)
    qc.rx(theta_1[0], 2)
    qc.ry(theta_1[1], 2)
    qc.rz(theta_1[2], 2)
    qc.barrier()

    # ── second half (acts on data-1 & ancilla) ─────────────────────────────
    qc.rx(theta_2[0], 1)
    qc.ry(theta_2[1], 1)
    qc.rz(theta_2[2], 1)
    qc.rx(theta_2[0], 2)
    qc.ry(theta_2[1], 2)
    qc.rz(theta_2[2], 2)
    qc.rxx(theta_2[3], 1, 2)
    qc.ryy(theta_2[4], 1, 2)
    qc.rzz(theta_2[5], 1, 2)
    qc.rx(theta_2[0], 1)
    qc.ry(theta_2[1], 1)
    qc.rz(theta_2[2], 1)
    qc.rx(theta_2[0], 2)
    qc.ry(theta_2[1], 2)
    qc.rz(theta_2[2], 2)
    qc.barrier()

    return qc


def syndrome_circuit(n: int, ancilla_params: np.ndarray) -> QuantumCircuit:
    """
    Prepare a syndrome circuit for `n` data qubits, using `n-1` ancillas.

    Args:
        n (int): Number of data qubits.
        ancilla_params (np.ndarray): List of parameters for the ancilla preparation circuits. Shape: (n-1, 2, 6).

    Returns:
        QuantumCircuit: A quantum circuit that prepares the syndrome state.
    """
    qc = QuantumCircuit(2 * n, name=f"syndrome_circuit_{n}")

    # Prepare the data qubits in a GHZ-like state |+...+⟩
    for i in range(n):
        qc.h(i)

    # Prepare the ancillas in pairs, each pair of data qubits
    for i in range(n - 1):
        ancilla_params_i = ancilla_params[i]
        ancilla_box = ancilla_prep(ancilla_params_i[0], ancilla_params_i[1])
        #qc.append(ancilla_box.to_instruction(), [i, i + 1, n + i])
        qc.compose(ancilla_box, [i, i + 1, n + i], inplace=True)
    return qc


def correction_circuit(n: int, param_1: np.ndarray, param_2: np.ndarray, param_3: np.ndarray) -> QuantumCircuit:
    """
    Three-shell recovery circuit, faithfully mirroring the TensorCircuit
    version (binary-tree fan-out  ➜  dense ancilla→data layer  ➜  per-qubit clean-up).

    Arg:
        n (int): Number of data qubits (n data + n ancilla).
        param_1 (np.ndarray): Parameters for the first shell of the correction circuit. Shape: (log2(n), n//2, 3).
        param_2 (np.ndarray): Parameters for the second shell of the correction circuit. Shape: (n, n, 2).
        param_3 (np.ndarray): Parameters for the third shell of the correction circuit. Shape: (n, 3).
    """
    corr = QuantumCircuit(2 * n, name="correction")

    # ---- shell 1: binary-tree fan-out over ancillas -----------------------
    m = int(np.log2(n))
    for i in range(m):
        count = 0
        for j in range(0, n, 2 ** (i + 1)):
            for k in range(2 ** i):
                corr.ry(np.pi / 2, n + j)
                corr.rxx(param_1[i][count][0], n + j, j + k + 2 ** i - 1)
                corr.rx(param_1[i][count][1], n + j)
                corr.rx(param_1[i][count][2], j + k + 2 ** i - 1)
                corr.ry(-np.pi / 2, n + j)
                count += 1
            # feed-forward CNOT in the fan-out tree
            if j + 2 ** i < n:
                corr.cx(n + j + 2 ** i, n + j)

        corr.barrier()

    # ---- shell 2: all-to-all ancilla→data layer ---------------------------
    for anc in range(n):
        for data in range(n):
            corr.ry(np.pi / 2, n + anc)
            corr.rxx(param_2[anc][data][0], n + anc, data)
            corr.rx(param_2[anc][data][1], n + anc)
            corr.ry(-np.pi / 2, n + anc)

    # ---- shell 3: per-data-qubit clean-up --------------------------------
    for q in range(n):
        corr.rx(param_3[q][0], q)
        corr.ry(param_3[q][1], q)
        corr.rz(param_3[q][2], q)

    return corr


def GHZ_circuit(n: int, params_flat: ParameterVector) -> QuantumCircuit:
    """
    Assemble the whole VQE ansatz on 2n wires:
        – Hadamards on data qubits
        – ZZ-syndrome layer
        – Correction circuit (three shells)
    A compact draw-friendly sub-circuit structure is used so the
    high-level diagram stays readable.

    Arg:
        n (int): Number of data qubits (n data + n ancilla).
        params_flat (ParameterVector): Flattened parameter vector.
    """
    # Define shapes and sizes to know how to slice and reshape
    shape_0 = (n - 1, 2, 6)
    shape_1 = (int(np.log2(n)), n // 2, 3)
    shape_2 = (n, n, 2)
    shape_3 = (n, 3)

    size_0 = int(np.prod(shape_0))
    size_1 = int(np.prod(shape_1))
    size_2 = int(np.prod(shape_2))

    # Define slice points
    end_0 = size_0
    end_1 = size_0 + size_1
    end_2 = size_0 + size_1 + size_2

    # --- Perform the slicing and reshaping ---
    param_0 = np.array(params_flat[0:end_0]).reshape(shape_0)
    param_1 = np.array(params_flat[end_0:end_1]).reshape(shape_1)
    param_2 = np.array(params_flat[end_1:end_2]).reshape(shape_2)
    param_3 = np.array(params_flat[end_2:]).reshape(shape_3)

    qc = QuantumCircuit(2 * n, name=f"VQE_QC_n{n}")
    synd = syndrome_circuit(n, param_0)
    #qc.append(synd.to_instruction(), qc.qubits)
    qc.compose(synd, inplace=True)

    corr = correction_circuit(n, param_1, param_2, param_3)
    #qc.append(corr.to_instruction(), qc.qubits)
    qc.compose(corr, inplace=True)

    return qc

def get_num_ghz_params(n: int) -> int:
    """
    Calculates the total number of trainable parameters for the GHZ_circuit.
    Args:
        n (int): The number of data qubits.
    Returns:
        int: The total number of parameters.
    """
    if n == 0:
        return 0
    if not (n > 0 and (n & (n - 1) == 0)):
        raise ValueError("n must be a power of 2 for the correction circuit structure.")

    shape_0 = (n - 1, 2, 6)
    shape_1 = (int(np.log2(n)), n // 2, 3)
    shape_2 = (n, n, 2)
    shape_3 = (n, 3)

    size_0 = np.prod(shape_0)
    size_1 = np.prod(shape_1)
    size_2 = np.prod(shape_2)
    size_3 = np.prod(shape_3)

    return int(size_0 + size_1 + size_2 + size_3)


if __name__ == "__main__":
    n = 8

    # Generate random parameters for the circuit
    param_0 = np.random.rand(n - 1, 2, 6).tolist()
    param_1 = np.random.rand(int(np.log2(n)), n // 2, 3).tolist()
    param_2 = np.random.rand(n, n, 2).tolist()
    param_3 = np.random.rand(n, 3).tolist()
    all_params = []
    all_params.extend(np.array(param_0).flatten())
    all_params.extend(np.array(param_1).flatten())
    all_params.extend(np.array(param_2).flatten())
    all_params.extend(np.array(param_3).flatten())

    # Convert to Qiskit ParameterVector
    params_flat = ParameterVector('p', len(all_params))

    qc = GHZ_circuit(n, params_flat)

    # ASCII/text preview (quick & dependency-free)
    print(qc.draw("text", fold=120))

    # Pretty matplotlib diagram (requires matplotlib)
    fig = qc.draw("mpl", fold=100)
    fig.suptitle(f"Full Variational Circuit (n = {n})", fontsize=14)
    fig.tight_layout()
    plt.show()
