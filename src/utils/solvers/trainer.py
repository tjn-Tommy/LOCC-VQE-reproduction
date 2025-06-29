import torch
import numpy as np
import time
import matplotlib.pyplot as plt


class BatchedVQE:
    """
    A class to run multiple VQE instances in parallel using PyTorch batching,
    configurable for CPU or GPU execution.
    """

    def __init__(self, cost_function, initial_params_batch, optimizer_name='Adam', learning_rate=0.1, device='cpu'):
        self.batch_size = initial_params_batch.shape[0]
        self.num_params = initial_params_batch.shape[1]
        self.device = device

        self.cost_function = cost_function
        self.params = torch.tensor(
            initial_params_batch,
            requires_grad=True,
            dtype=torch.float32,
            device=self.device,
        )

        optimizer_class = getattr(torch.optim, optimizer_name)
        self.optimizer = optimizer_class([self.params], lr=learning_rate)

        self.history = {'cost': [[] for _ in range(self.batch_size)]}
        self.final_energies = None

    def optimize(self, iterations, verbose=False):
        """Runs the batched VQE optimization loop."""
        print(f"Starting batched VQE optimization on device: '{self.device}' for {self.batch_size} instances...")
        start_time = time.time()

        for i in range(iterations):
            self.optimizer.zero_grad()
            cost_batch = self.cost_function(self.params)
            total_cost = cost_batch.sum()
            total_cost.backward()
            self.optimizer.step()

            if verbose and (i + 1) % 1 == 0:
                avg_cost = cost_batch.mean().item()
                print(f"Iteration {i + 1:5d} | Average Cost: {avg_cost:.8f}")

            for b in range(self.batch_size):
                self.history['cost'][b].append(cost_batch[b].item())

        end_time = time.time()
        print(f"\nOptimization finished in {end_time - start_time:.2f} seconds.")

        self.final_energies = self.params.grad.new_tensor([run_history[-1] for run_history in self.history['cost']])
        return self.final_energies

    def analyze_results(self, exact_energy=None):
        """Calculates and prints statistics of the final energies."""
        if self.final_energies is None:
            print("Please run optimize() first.")
            return

        # Move data to CPU for analysis with NumPy and Matplotlib
        final_energies_cpu = self.final_energies.cpu().detach().numpy()

        mean_energy = np.mean(final_energies_cpu)
        std_energy = np.std(final_energies_cpu)
        min_energy = np.min(final_energies_cpu)
        max_energy = np.max(final_energies_cpu)

        print("\n--- VQE Performance Analysis ---")
        print(f"Number of parallel runs: {self.batch_size}")
        print(f"Best energy found:       {min_energy:.8f}")
        print(f"Worst energy found:      {max_energy:.8f}")
        print(f"Average final energy:    {mean_energy:.8f}")
        print(f"Std. Dev. of energies:   {std_energy:.8f}")

        if exact_energy is not None:
            print("----------------------------------")
            print(f"Exact ground state energy: {exact_energy:.8f}")
            print(f"Error of best result:    {abs(min_energy - exact_energy):.8f}")

        plt.figure(figsize=(12, 6))
        plt.hist(final_energies_cpu, bins=15, alpha=0.75, label='Final Energy Distribution')
        plt.axvline(mean_energy, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {mean_energy:.4f}')
        plt.axvline(min_energy, color='green', linestyle='dashed', linewidth=2, label=f'Best: {min_energy:.4f}')
        if exact_energy is not None:
            plt.axvline(exact_energy, color='black', linestyle='solid', linewidth=2, label=f'Exact: {exact_energy:.4f}')

        plt.title('Distribution of Final Energies from Batched VQE Runs')
        plt.xlabel('Final Energy')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(axis='y', alpha=0.5)
        plt.show()

