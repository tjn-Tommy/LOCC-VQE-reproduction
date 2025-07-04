from collections.abc import Callable
from jax import config
# Must happen before any JAX imports
config.update("jax_enable_x64", True)
import flax.linen as nn
import tensorcircuit as tc
import jax
import jax.numpy as jnp
import math
import jax.nn as jnn
from functools import partial
from typing import Any
from flax.core import freeze, unfreeze
from flax.nnx import update
from flax.traverse_util import flatten_dict, unflatten_dict
import optax

tc.set_backend("jax")

def uvqe_factory(
                 params: jnp.ndarray,
                 n: int,
                 circ: Callable[[jnp.ndarray], tc.Circuit],
                 hamiltonian: Callable[[tc.Circuit, int], jnp.ndarray],
                 ctype: jnp.dtype = jnp.complex128,
                 htype: jnp.dtype = jnp.float64,
                 ) -> jnp.ndarray:
    """
    Factory function to create the adaptive VQE circuit and compute the energy.
    Args:
        params: jnp.ndarray, parameters for the ansatz.
        n: int, number of qubits.
        circ: Callable, function to create the circuit.
        hamiltonian: Callable, function to compute the Hamiltonian expectation.
        ctype: jnp.dtype, data type for complex numbers.
        htype: jnp.dtype, data type for Hamiltonian.
    """
    # cast into complex dtype
    params = params.astype(ctype)
    circuit = circ(params)
    energy = hamiltonian(circuit, n)
    return energy


def grad_params_paramshift(
        n_bits: int,
        circ: Callable[[jnp.ndarray], tc.Circuit],
        params: jnp.ndarray,
        hamiltonian: Callable[[tc.Circuit, int], jnp.ndarray],
        ctype: jnp.dtype = jnp.complex128,
        htype: jnp.dtype = jnp.float64,
        ftype: jnp.dtype = jnp.float64,
):

    # --------------------- 1. factory for helper functions -------------------
    # params: params
    uvqe = partial(uvqe_factory,
                            n=n_bits,
                            circ=circ,
                            hamiltonian=hamiltonian,
                            ctype=ctype,
                            htype=htype)
    uvqe_vmap = jax.vmap(uvqe, in_axes=(0,), out_axes=0)
    # ---------------------- 2. Initialize variables and tensors ----------------
    batch_size = params.shape[0]
    params_num = params.shape[1]

    # build parameter-shifted theta_1
    params_full = jnp.tile( params[:, None, :], (1, params_num, 1)).astype(ftype) # (batch_size, params_num, params_num)

    shift_tensor = (jnp.pi / 2) * jnp.eye(params_num, dtype=ftype)
    shift_tensor = jnp.tile( shift_tensor[None,  :, :], (batch_size, 1, 1)) # (batch_size, params_num, params_num)

    # ---- (+) shift -----------------------------------------------------------
    params_pos = (params_full + shift_tensor).reshape(-1, params_num)
    energy_pos = uvqe_vmap(params_pos).reshape(-1, params_num)

    # ---- (–) shift -----------------------------------------------------------
    params_neg = (params_full - shift_tensor).reshape(-1, params_num)
    energy_neg = uvqe_vmap(params_neg).reshape(-1, params_num)

    # ----- gradient computation ---------------------------------------------------
    grad_params = 0.5 * (energy_pos - energy_neg)

    return grad_params

def train_step(
    n_bits: int,
    circ: Callable[[jnp.ndarray], tc.Circuit],
    params: jnp.ndarray,
    hamiltonian: Callable[[tc.Circuit, int], jnp.ndarray],
    optimizer: Any,
    optimizer_state: Any,
    ctype: jnp.dtype = jnp.complex128,
    htype: jnp.dtype = jnp.float64,
    ftype: jnp.dtype = jnp.float64,
):
    """
    Perform a single training step for the adaptive VQE.
    Args:
        model: Flax nn.Module, the model to optimize.
        unravel: Callable, function to convert model parameters to a flat array.
        n_bits: int, number of system qubits.
        synd: Callable, function to apply the syndrome circuit.
        theta_1: jnp.ndarray, angles for the first ansatz.
        corr: Callable, function to apply the correlation circuit.
        hamiltonian: Callable, function to compute the Hamiltonian expectation.
        gamma: jnp.ndarray, parameters for the second ansatz.
        sample_round: int, number of sampling rounds.
        optimizer: optax.GradientTransformation, optimizer to use.
        optimizer_state: optax.OptState, state of the optimizer.
        rootkey: jax.random.PRNGKey, random key for sampling.
        ctype: jnp.dtype, data type for complex numbers.
        htype: jnp.dtype, data type for Hamiltonian.
        ftype: jnp.dtype, data type for floating point numbers.
    Returns:
        updates: jnp.ndarray, the updates to apply to the model parameters.
        optimizer_state: optax.OptState, updated state of the optimizer.
    """
    # Compute the gradient with respect to theta_1
    grad_params= grad_params_paramshift(
        n_bits=n_bits,
        circ=circ,
        params=params,
        hamiltonian=hamiltonian,
        ctype=ctype,
        htype=htype,
        ftype=ftype,
    )

    # Combine parameters and gradients
    opt_params  = {
        'params': params,
    }
    opt_grads = {
        'params': grad_params,
    }
    # Update the optimizer state
    update_vmap = jax.vmap(optimizer.update, in_axes=(0, 0, 0), out_axes=(0, 0))
    updates, optimizer_state = update_vmap(opt_grads, optimizer_state, opt_params)
    new_params = optax.apply_updates(opt_params, updates)
    mean_grad_theta1 = jnp.mean(grad_params)
    return new_params, optimizer_state, mean_grad_theta1

def energy_estimator(
        n_bits: int,
        circ: Callable[[jnp.ndarray], tc.Circuit],
        params: jnp.ndarray,
        hamiltonian: Callable[[tc.Circuit, int], jnp.ndarray],
        ctype: jnp.dtype = jnp.complex128,
        htype: jnp.dtype = jnp.float64,
        ftype: jnp.dtype = jnp.float64,
):

    # --------------------- 1. factory for helper functions -------------------
    # params: params
    uvqe = partial(uvqe_factory,
                            n=n_bits,
                            circ=circ,
                            hamiltonian=hamiltonian,
                            ctype=ctype,
                            htype=htype)
    uvqe_vmap = jax.vmap(uvqe, in_axes=(0,), out_axes=0)
    # ---------------------- 2. Initialize variables and tensors ----------------
    energy = uvqe_vmap(params)
    min_energy = jnp.min(energy)
    mean_energy = jnp.mean(energy)
    return min_energy, mean_energy
