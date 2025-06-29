import qiskit
import torch
import torch.nn as nn
from qiskit.primitives import StatevectorEstimator
from qiskit.primitives import BackendEstimatorV2
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit import QuantumCircuit
from qiskit_machine_learning.connectors import TorchConnector
from qiskit_machine_learning.gradients import ParamShiftEstimatorGradient

def qnn_gen(
    circuit: qiskit.QuantumCircuit,
    observables: list,
    estimator,
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
        estimator (BackendEstimatorV2, optional): The estimator to use for evaluating the circuit.

    Returns:
        EstimatorQNN: A Qiskit EstimatorQNN instance.
    """
    return EstimatorQNN(
        circuit=circuit,
        observables=observables,
        weight_params=weight_params,
        input_params=input_params,
        estimator= estimator,
        gradient=ParamShiftEstimatorGradient(estimator)
    )


def qnn_wrapper (
    circuit: qiskit.QuantumCircuit,
    observables: list,
    estimator,
    input_params: list = None
) -> TorchConnector:
    """
    Wraps a Qiskit EstimatorQNN in a TorchConnector for PyTorch compatibility.

    Args:
        circuit (QuantumCircuit): The quantum circuit to use.
        observables (list): List of observables to measure.
        input_params (list, optional): List of input parameters for the circuit.
        estimator (BackendEstimatorV2): The estimator to use for evaluating the circuit.
    Returns:
        TorchConnector: A TorchConnector instance wrapping the EstimatorQNN.
    """
    origin_qnn = qnn_gen(circuit, observables, estimator, list(circuit.parameters), input_params)
    return TorchConnector(origin_qnn)
