import qiskit
import torch
import torch.nn as nn
from qiskit.primitives import StatevectorEstimator as Estimator
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit import QuantumCircuit
from qiskit_machine_learning.connectors import TorchConnector

def qnn_gen(
    circuit: qiskit.QuantumCircuit,
    observables: list,
    weight_params: list = None,
    input_params: list = None
) -> EstimatorQNN:
    """
    Generates a Qiskit EstimatorQNN from a quantum circuit and observables.

    Args:
        circuit (QuantumCircuit): The quantum circuit to use.
        observables (list): List of observables to measure.
        weight_params (list, optional): List of weight parameters for the circuit.
        input_params (list, optional): List of input parameters for the circuit.

    Returns:
        EstimatorQNN: A Qiskit EstimatorQNN instance.
    """
    est = Estimator()
    return EstimatorQNN(
        circuit=circuit,
        observables=observables,
        weight_params=weight_params,
        input_params=input_params,
        estimator= est,
    )


def qnn_wrapper (
    circuit: qiskit.QuantumCircuit,
    observables: list,
    input_params: list = None,
) -> TorchConnector:
    """
    Wraps a Qiskit EstimatorQNN in a TorchConnector for PyTorch compatibility.

    Args:
        circuit (QuantumCircuit): The quantum circuit to use.
        observables (list): List of observables to measure.
        input_params (list, optional): List of input parameters for the circuit.
    Returns:
        TorchConnector: A TorchConnector instance wrapping the EstimatorQNN.
    """
    origin_qnn = qnn_gen(circuit, observables, list(circuit.parameters), input_params)
    return TorchConnector(origin_qnn, initial_weights=torch.zeros(origin_qnn.num_weights))




