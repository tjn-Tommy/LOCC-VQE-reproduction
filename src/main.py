import os
#os.environ["JAX_TRACEBACK_FILTERING"] = "off"
os.environ["JAX_ENABLE_X64"] = "0"
import jax
import yaml
import time
import matplotlib.pyplot as plt
import datetime
import tensorcircuit as tc
import numpy as np
from jax import numpy as jnp
from solvers import UVQESolver, LOCCSolver, NonSampleSolver

def main():
    with open("./configs/ns_config.yaml", 'r') as f:
        ns_config = yaml.safe_load(f)
    ns_solver = NonSampleSolver(ns_config)
    #ns_solver.train(2000, 0.1)
    ns_results = {}
    for i in range(11):
        result = ns_solver.train(1000, 0.1 * i)
        ns_results[i] = result

    with open("./configs/jax_config.yaml", 'r') as f:
        locc_config = yaml.safe_load(f)
    locc_solver = LOCCSolver(locc_config)
    #locc_solver.train(1000, 0.1)
    results_locc = {}
    for i in range(11):
        result = locc_solver.train(1000, 0.1 * i)
        results_locc[i] = result

    with open("./configs/uvqe_config.yaml", 'r') as f:
        uvqe_config = yaml.safe_load(f)
    uvqe_solver = UVQESolver(uvqe_config)
    results = {}
    for i in range(11):
        result = uvqe_solver.train(1000, 0.1 * i)
        results[i] = result

    # Prepare data for plotting
    perturb = [0.1 * i for i in ns_results.keys()]
    exact_energy = [ns_results[i]['exact_energy'] for i in ns_results.keys()]
    final_min_energy = [results[i]['mean_min_energy'] for i in results.keys()]
    final_min_energy_locc = [results_locc[i]['mean_min_energy'] for i in results_locc.keys()]
    final_min_energy_ns = [ns_results[i]['mean_min_energy'] for i in ns_results.keys()]
    out_dir = os.path.join("results", ns_solver.exp_name, ns_solver.timestamp_str)
    # Create the plot
    plt.style.use('seaborn-v0_8-darkgrid')
    plt.plot(perturb, exact_energy, label='Exact Energy')
    plt.plot(perturb, final_min_energy, label='Final Min Energy (Unitary)')
    plt.plot(perturb, final_min_energy_ns, label='Final Min Energy (LOCC-Non-Sampling)')
    plt.plot(perturb, final_min_energy_locc, label='Final Min Energy (LOCC-Sample)')
    plt.xlabel('Perturb (0.1 × i)')
    plt.ylabel('Energy')
    plt.title('Energy vs. Perturbation')
    plt.legend()
    plt.savefig(os.path.join(out_dir, 'mean_grad_plot.png'))
    plt.show()


if __name__ == '__main__':
    main()
