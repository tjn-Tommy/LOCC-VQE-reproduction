import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import yaml
import argparse
import logging
from pathlib import Path
import time

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- The Trainer Class ---
import torch
import numpy as np
import time
import matplotlib.pyplot as plt


class PyTorchVQE:
    """
    A class to run the Variational Quantum Eigensolver (VQE) using PyTorch.

    This class abstracts the optimization loop, allowing it to be used with any
    differentiable quantum cost function (e.g., from PennyLane or Qiskit's TorchConnector).
    """

    def __init__(self, cost_function, initial_params, optimizer_name='Adam', learning_rate=0.1):
        """
        Initializes the VQE solver.

        Args:
            cost_function (callable): A function that takes a PyTorch tensor of
                parameters and returns a scalar loss (expectation value).
                This function must be differentiable with respect to its input.
            initial_params (np.ndarray or list): The starting parameters for the VQE ansatz.
            optimizer_name (str): The name of the PyTorch optimizer to use.
                Supported: 'Adam', 'SGD', 'RMSprop'.
            learning_rate (float): The learning rate for the optimizer.
        """
        if not callable(cost_function):
            raise TypeError("cost_function must be a callable function.")

        self.cost_function = cost_function
        self.params = torch.tensor(initial_params, requires_grad=True, dtype=torch.float64)

        self.learning_rate = learning_rate
        if optimizer_name.lower() == 'adam':
            self.optimizer = torch.optim.Adam([self.params], lr=self.learning_rate)
        elif optimizer_name.lower() == 'sgd':
            self.optimizer = torch.optim.SGD([self.params], lr=self.learning_rate)
        elif optimizer_name.lower() == 'rmsprop':
            self.optimizer = torch.optim.RMSprop([self.params], lr=self.learning_rate)
        else:
            raise ValueError(f"Optimizer '{optimizer_name}' not supported.")

        self.history = {'cost': []}
        self.final_energy = None
        self.optimal_params = None

    def optimize(self, iterations, verbose=True):
        """
        Runs the VQE optimization loop.
        """
        print("Starting PyTorch VQE optimization with Qiskit backend...")
        start_time = time.time()

        for i in range(iterations):
            self.optimizer.zero_grad()
            cost = self.cost_function(self.params)
            cost.backward()
            self.optimizer.step()

            cost_val = cost.item()
            self.history['cost'].append(cost_val)

            if verbose and (i + 1) % 10 == 0:
                print(f"Iteration {i + 1:5d} | Cost: {cost_val:.8f}")

        end_time = time.time()
        print(f"\nOptimization finished in {end_time - start_time:.2f} seconds.")

        self.final_energy = self.history['cost'][-1]
        self.optimal_params = self.params.detach().numpy()

        print(f"Final Energy (Expectation Value): {self.final_energy:.8f}")

        return {'fun': self.final_energy, 'x': self.optimal_params}

    def plot_history(self):
        """
        Plots the cost history of the optimization.
        """
        if not self.history['cost']:
            print("No optimization history to plot. Please run optimize() first.")
            return

        plt.figure(figsize=(10, 6))
        plt.plot(self.history['cost'], label='Cost (Energy)')
        plt.xlabel("Iteration")
        plt.ylabel("Expectation Value")
        plt.title("VQE Optimization History (Qiskit + PyTorch)")
        plt.grid(True)
        plt.legend()
        plt.show()