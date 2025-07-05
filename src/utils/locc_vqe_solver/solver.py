from collections.abc import Callable
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
from .helpers import convert_ndarray_to_params, forward_pass, jacobian_wrt_params, make_batch_keys
from .sampling import get_prob, sample_factory

tc.set_backend("jax")

def adaptive_vqe_factory(
                 theta_1: jnp.ndarray,
                 theta_2: jnp.ndarray,
                 projector_onehot: jnp.ndarray,
                 sample_cond_prob: jnp.ndarray,
                 n: int,
                 synd: Callable[[int, jnp.ndarray], tc.Circuit],
                 corr: Callable[[tc.Circuit, int, jnp.ndarray], tc.Circuit],

                 hamiltonian: Callable[[tc.Circuit, int], jnp.ndarray],
                 ctype: jnp.dtype = jnp.complex64,
                 htype: jnp.dtype = jnp.float32,
                 ) -> jnp.ndarray:
    """
    n_bits:      number of system qubits
    theta_1:           shape ( ) or scalar angle for U1
    theta_2:           shape ( ) or scalar angle for U2
    projector_onehot:  shape (n_bits, 3) one-hot selectors per qubit
    sample_cond_prob:  scalar (the product of conditional probs)
    kx, h:             extra args passed through to Hamiltonian(...)
    returns:           normalized energy expectation
    """
    # cast into complex dtype
    theta_1 = theta_1.astype(ctype)
    theta_2 = theta_2.astype(ctype)
    projector_onehot = projector_onehot.astype(ctype)
    circuit = synd(n, theta_1)


    # predefine the 3 projectors: |0⟩⟨0|, |1⟩⟨1|, I
    projector_set = jnp.array([
        [[1., 0.], [0., 0.]],  # zero projector
        [[0., 0.], [0., 1.]],  # one projector
        [[1., 0.], [0., 1.]]  # identity
    ], dtype=ctype)  # shape (3,2,2)

    # apply each projector to its ancilla line
    for idx in range(n):
        # select the one-hot for qubit idx: shape (3,)
        sel = projector_onehot[idx]  # (3,)
        # compute projector unitary for this qubit
        proj_unit = jnp.sum(sel[:, None, None] * projector_set, axis=0)
        # then insert it on the ancilla line n_bits+idx
        circuit.any(n + idx, unitary=proj_unit)
    # append the second ansatz
    corr(circuit, n, theta_2)

    # measure energy and normalize by the conditional-sample prob
    energy = hamiltonian(circuit, n) / sample_cond_prob.astype(htype)
    return energy


def grad_theta_1_paramshift_sample(
        n_bits: int,
        unravel: Callable[[jax.Array], dict],
        synd: Callable[[tc.Circuit, int, jnp.ndarray], tc.Circuit],
        theta_1_batched: jnp.ndarray, # shape (batch_size, theta_1_num)
        corr: Callable[[tc.Circuit, int, jnp.ndarray], tc.Circuit],
        model: nn.Module,
        gamma_batched: jnp.ndarray,
        hamiltonian: Callable[[tc.Circuit, int], jnp.ndarray],
        sample_round: int,
        input_key: jax.random.PRNGKey,
        ctype: jnp.dtype = jnp.complex64,
        htype: jnp.dtype = jnp.float32,
        ftype: jnp.dtype = jnp.float32,
):

    # --------------------- 1. factory for helper functions -------------------
    # params: theta_1, theta_2, projector_onehot, sample_cond_prob
    adaptive_vqe = partial(adaptive_vqe_factory,
                            n=n_bits,
                            synd=synd,
                            corr=corr,
                            hamiltonian=hamiltonian,
                            ctype=ctype,
                            htype=htype)
    adaptive_vqe_vmap = jax.vmap(adaptive_vqe, in_axes=(0, 0, 0, 0), out_axes=0)
    # params: randkey, theta_1
    sample = partial(sample_factory,
                        generator=synd,
                        n_bits=n_bits)
    sample_vmap = jax.vmap(sample, in_axes=(0, 0), out_axes=(0, 0))
    # ---------------------- 2. Initialize variables and tensors ----------------
    batch_size = theta_1_batched.shape[0]
    theta1_num = theta_1_batched.shape[1]

    # build parameter-shifted theta_1
    theta_1_full = jnp.tile( theta_1_batched[:, None, None, :], (1, sample_round, theta1_num, 1)).astype(ftype) # (batch_size, sample_round, θ₁_num, θ₁_num)

    shift_tensor = (jnp.pi / 2) * jnp.eye(theta1_num, dtype=ftype)
    shift_tensor = jnp.tile( shift_tensor[None, None, :, :], (batch_size, sample_round, 1, 1)) # (batch_size, sample_round, θ₁_num, θ₁_num)

    # ---- 3. Compute the energy using parameter-shift rule -----------------------
    def _energy(theta_1_shifted, gamma, randkey):
        """
                    Args:
                        theta_1_shifted      : shape (batch, sample_round, θ₁_num, θ₁_num)
                        gamma                : shape (batch, sample_round, θ₁_num, γ_num)
                        randkey              : PRNGKey for sampling, shape (batch, sample_round, θ₁_num, 2)
                    """
        # Sample projectors and conditional probabilities
        theta_1_reshaped = jnp.reshape(theta_1_shifted,
                                       (-1, theta1_num))  # (batch_size * sample_round * θ₁_num, θ₁_num)
        gamma_reshaped = jnp.reshape(gamma,
                                     (batch_size, gamma_batched.shape[-1]))  # (batch_size, γ_num)
        gamma_converted = unravel(gamma_reshaped)  # convert to params dict for model
        randkey_reshaped = jnp.reshape(randkey, (-1, 2))  # (batch_size, sample_round * theta1_num, 2)

        projector, cond_prob = sample_vmap(randkey_reshaped, theta_1_reshaped)  # shape (batch_size * sample_round * θ₁_num, n_bits), shape (batch_size * sample_round * θ₁_num,)
        measure_result = 2 * (projector - 0.5)
        # one-hot encoding of projectors
        proj_onehot = jnn.one_hot(projector, 3)  # shape (batch_size * sample_round * theta1_num, n_bits, 3)
        forward_pass_vmap = jax.vmap(forward_pass, in_axes=(None, 0, 0), out_axes=0)
        theta_2 = forward_pass_vmap(
            model,
            measure_result.reshape(batch_size, -1, n_bits),  # shape (batch_size, sample_round * θ₁_num, n_bits)
            gamma_converted
        )  # shape (batch_size, sample_round * θ₁_num, θ₂_num)
        # params: theta_1, theta_2, projector_onehot, sample_cond_prob
        energy = adaptive_vqe_vmap(
            theta_1_reshaped,
            theta_2.reshape(batch_size * sample_round * theta1_num, -1),
            proj_onehot,
            cond_prob) # shape (batch_size * sample_round * θ₁_num,)

        # Reshape back to (batch_size, sample_round, θ₁_num)
        energy = jnp.reshape(energy, (batch_size, sample_round, theta1_num))
        return energy

    # ---- (+) shift -----------------------------------------------------------
    theta_1_pos = theta_1_full + shift_tensor # (batch_size, sample_round, θ₁_num, θ₁_num)
    root_key, batch_keys = make_batch_keys(input_key, batch_size * sample_round * theta1_num)
    energy_pos = _energy(theta_1_pos, gamma_batched, batch_keys) # (batch_size, sample_round, θ₁_num)

    # ---- (–) shift -----------------------------------------------------------
    theta_1_neg = theta_1_full - shift_tensor
    root_key, batch_keys = make_batch_keys(root_key, batch_size * sample_round * theta1_num)
    energy_neg = _energy(theta_1_neg, gamma_batched, batch_keys)

    # ----- gradient computation ---------------------------------------------------
    grad_theta_1_round = 0.5 * (energy_pos - energy_neg)
    # calculate mean and variance across sample rounds
    grad_theta_1 = jnp.mean(grad_theta_1_round, axis=1)  # batch, θ₁_num
    var_grad = jnp.var(grad_theta_1_round, axis=1)  # batch, θ₁_num

    return grad_theta_1, var_grad

def grad_gamma_batched(
        n_bits: int,
        unravel: Callable[[jax.Array], dict],
        synd: Callable[[tc.Circuit, int, jnp.ndarray], tc.Circuit],
        theta_1_batched: jnp.ndarray,  # shape (batch_size, theta_1_num)
        corr: Callable[[tc.Circuit, int, jnp.ndarray], tc.Circuit],
        model: nn.Module,
        gamma_batched: jnp.ndarray,
        hamiltonian: Callable[[tc.Circuit, int], jnp.ndarray],
        sample_round: int,
        input_key: Any,
        ctype: jnp.dtype = jnp.complex128,
        htype: jnp.dtype = jnp.float64,
        ftype: jnp.dtype = jnp.float64,
):
    """
    Compute the gradient of the energy with respect to theta_2 using parameter-shift rule.
    """
    # --------------------- 1. factory for helper functions -------------------
    # params: theta_1, theta_2, projector_onehot, sample_cond_prob
    adaptive_vqe = partial(adaptive_vqe_factory,
                            n=n_bits,
                           synd=synd,
                            corr=corr,
                            hamiltonian=hamiltonian,
                            ctype=ctype,
                            htype=htype)
    adaptive_vqe_vmap = jax.vmap(adaptive_vqe, in_axes=(0, 0, 0, 0), out_axes=0)
    # params: randkey, theta_1
    sample = partial(sample_factory,
                        generator=synd,
                        n_bits=n_bits)
    sample_vmap = jax.vmap(sample, in_axes=(0, 0), out_axes=(0, 0))
    unravel_vmap = jax.vmap(unravel, in_axes=(0,), out_axes=0)
    # ---------------------- 2. Initialize variables and tensors ----------------
    batch_size = theta_1_batched.shape[0]
    theta1_num = theta_1_batched.shape[1]
    gamma_num = gamma_batched.shape[-1] # shape (batch_size, gamma_num)
    dummy_theta_1 = jnp.zeros((theta1_num,), dtype=ftype)  # dummy theta_1 for shape
    dummy_gamma = jnp.zeros((gamma_num,), dtype=ftype)  # dummy gamma for shape
    def _theta_2_generator(theta_1, gamma, randkey):
        """
        Generate theta_2 based on the shifted theta_1 and gamma.
        """
        projector, cond_prob = sample_vmap(randkey, theta_1)  # shape (batch_size, n_bits), shape (batch_size,)
        measure_result = 2 * (projector - 0.5) # convert to {-1, 1} for projectors
        # one-hot encoding of projectors
        proj_onehot = jnn.one_hot(projector, 3)  # shape (batch_size, n_bits, 3)
        params = unravel(gamma)
        theta_2 = forward_pass(model, measure_result, params)  # shape (batch_size, theta_2_num)
        return theta_2, proj_onehot, cond_prob, measure_result

    # dummy theta_2 for get the size of the output
    root_key, dummy_key = jax.random.split(input_key)
    _theta_2_generator_vmap = jax.vmap(_theta_2_generator, in_axes=(0, 0, 0), out_axes=(0, 0, 0, 0))
    dummy_theta_2, _, _, _ = _theta_2_generator(dummy_theta_1[None, :], dummy_gamma, dummy_key[None, :])
    theta2_num = dummy_theta_2.shape[-1]  # number of parameters in theta_2

    # ---- 3. Generate parameter-shifted theta_2 -----------------------
    root_key, batch_keys = make_batch_keys(root_key, batch_size * sample_round)
    # theta_2_batched shape (batch_size * sample_round, θ₂_num), projector_onehot_batched shape (batch_size * sample_round, n_bits, 3), sample_cond_prob_batched shape (batch_size * sample_round, )
    theta_2_batched, projector_onehot_batched, sample_cond_prob_batched, measure_result_batched = \
        _theta_2_generator_vmap(jnp.tile(theta_1_batched[:, None, :], (1, sample_round, 1)),  # (batch_size, sample_round, θ₁_num)
                            gamma_batched.reshape(-1, gamma_num),  # (batch_size, γ_num)
                                batch_keys.reshape(-1,sample_round, 2)  # (batch_size, sample_round, θ₂_num)
                                )
    theta_2_batched = theta_2_batched.reshape(-1, theta2_num)
    projector_onehot_batched = projector_onehot_batched.reshape(-1, n_bits, 3)  # (batch_size * sample_round * θ₂_num, n_bits, 3)
    sample_cond_prob_batched = sample_cond_prob_batched.reshape(-1)  # (batch_size * sample_round * θ₂_num, )
    measure_result_batched = measure_result_batched.reshape(-1, n_bits)  # (batch_size * sample_round * θ₂_num, n_bits)
    shift_tensor = (jnp.pi / 2) * jnp.eye(theta2_num, dtype=ftype)
    shift_tensor_batched_tiled = jnp.tile(shift_tensor[None,  :, :], (batch_size * sample_round, 1, 1))  # (batch_size * sample_round, θ₂_num, θ₂_num)
    theta_2_batched_tiled = jnp.tile(theta_2_batched[:, None, :], (1, theta2_num, 1)).astype(ftype)
    theta_2_pos = (theta_2_batched_tiled + shift_tensor_batched_tiled).reshape(-1, theta2_num)
    theta_2_neg = (theta_2_batched_tiled - shift_tensor_batched_tiled).reshape(-1, theta2_num)  # (batch_size * sample_round * θ₂_num, θ₂_num)

    theta_1_batched_tiled = jnp.tile(theta_1_batched[:, None, None, :], (1, sample_round, theta2_num, 1)).reshape(-1, theta1_num).astype(ftype)  # (batch_size * sample_round * θ₂_num, θ₁_num)
    projector_onehot_tiled = jnp.tile(projector_onehot_batched[:, None, :, :], (1, theta2_num, 1, 1)).reshape(-1, n_bits, 3).astype(ftype)  # (batch_size * sample_round * θ₂_num, n_bits, 3)
    sample_cond_prob_tiled = jnp.tile(sample_cond_prob_batched[:, None], (1, theta2_num)).reshape(-1)  # (batch_size * sample_round * θ₂_num, )
    # ---- 4. Compute the energy using parameter-shift rule -----------------------
    # params: theta_1, theta_2, projector_onehot, sample_cond_prob
    pos_energy = adaptive_vqe_vmap(
        theta_1_batched_tiled,
        theta_2_pos,
        projector_onehot_tiled,
        sample_cond_prob_tiled
    )  # (batch_size * sample_round * θ₂_num, )

    neg_energy = adaptive_vqe_vmap(
        theta_1_batched_tiled,
        theta_2_neg,
        projector_onehot_tiled,
        sample_cond_prob_tiled
    )  # (batch_size * sample_round * θ₂_num, )

    # ----- gradient computation ---------------------------------------------------
    grad_theta_2_round = 0.5 * (pos_energy - neg_energy)  # (batch_size * sample_round * θ₂_num, )
    grad_theta_2 = grad_theta_2_round.reshape(batch_size, sample_round, theta2_num)  # (batch_size, sample_round, θ₂_num)
    jacobian_vmap = jax.vmap(jacobian_wrt_params, in_axes=(None, 0, 0), out_axes=0)
    jacobian = jacobian_vmap(
        model,
        measure_result_batched.reshape(batch_size, sample_round, n_bits), # (batch_size, sample_round, n_bits)
        unravel_vmap(gamma_batched)  # (batch_size, γ_num)
    )  # (batch_size, sample_round, theta2_num, gamma_num)
    # Compute the gradient with respect to gamma
    gamma_grads_per_sample = jnp.einsum(
        'b s t g, b s t -> b s g',
        jacobian,
        grad_theta_2
    )
    grad_var = jnp.var(gamma_grads_per_sample, axis=1)  # (batch_size, θ₂_num)
    gamma_grads = gamma_grads_per_sample.sum(axis=1)
    return gamma_grads, grad_var

def train_step(
    model: nn.Module,
    unravel: Callable,
    n_bits: int,
    synd: Callable[[tc.Circuit, int, jnp.ndarray], tc.Circuit],
    theta_1: jnp.ndarray,
    corr: Callable[[tc.Circuit, int, jnp.ndarray], tc.Circuit],
    hamiltonian: Callable[[tc.Circuit, int], jnp.ndarray],
    gamma: jnp.ndarray,
    theta1_sample_round: int,
    gamma_sample_round: int,
    optimizer: Any,
    optimizer_state: Any,
    rootkey: jax.random.PRNGKey = jax.random.PRNGKey(0),
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
    unravel_vmap = jax.vmap(unravel, in_axes=(0,), out_axes=0)
    rootkey, subkey = jax.random.split(rootkey, 2)
    grad_theta_1, grad_var_theta1= grad_theta_1_paramshift_sample(
        n_bits=n_bits,
        unravel =unravel_vmap,
        synd=synd,
        theta_1_batched=theta_1,
        corr=corr,
        model=model,
        gamma_batched=gamma,
        hamiltonian=hamiltonian,
        sample_round=theta1_sample_round,
        input_key=subkey,
        ctype=ctype,
        htype=htype,
        ftype=ftype,
    )
    rootkey, subkey = jax.random.split(rootkey, 2)
    # Compute the gradient with respect to gamma
    grad_gamma, grad_var_gamma = grad_gamma_batched(
        n_bits=n_bits,
        unravel=unravel,
        synd=synd,
        theta_1_batched=theta_1,
        corr=corr,
        model=model,
        gamma_batched=gamma,
        hamiltonian=hamiltonian,
        sample_round=gamma_sample_round,
        input_key=subkey,
        ctype=ctype,
        htype=htype,
        ftype=ftype,
    )

    # Combine parameters and gradients
    opt_params  = {
        'theta_1': theta_1,
        'gamma': gamma
    }
    opt_grads = {
        'theta_1': grad_theta_1,
        'gamma': grad_gamma
    }
    # Update the optimizer state
    update_vmap = jax.vmap(optimizer.update, in_axes=(0, 0, 0), out_axes=(0, 0))
    updates, optimizer_state = update_vmap(opt_grads, optimizer_state, opt_params)
    new_params = optax.apply_updates(opt_params, updates)
    mean_grad_theta1 = jnp.mean(jnp.abs(grad_theta_1))
    mean_grad_gamma = jnp.mean(jnp.abs(grad_gamma))
    mean_grad_var_theta1 = jnp.mean(grad_var_theta1)
    mean_grad_var_gamma = jnp.mean(grad_var_gamma)
    return new_params, optimizer_state, mean_grad_var_theta1, mean_grad_var_gamma, mean_grad_theta1, mean_grad_gamma

def energy_estimator(
        n_bits: int,
        unravel: Callable[[jax.Array], dict],
        synd: Callable[[int, jnp.ndarray], tc.Circuit],
        theta_1_batched: jnp.ndarray, # shape (batch_size, theta_1_num)
        corr: Callable[[tc.Circuit, int, jnp.ndarray], tc.Circuit],
        model: nn.Module,
        gamma_batched: jnp.ndarray,
        hamiltonian: Callable[[tc.Circuit, int], jnp.ndarray],
        sample_round: int,
        input_key: jax.random.PRNGKey,
        ctype: jnp.dtype = jnp.complex128,
        htype: jnp.dtype = jnp.float64,
        ftype: jnp.dtype = jnp.float64,
):

    # --------------------- 1. factory for helper functions -------------------
    # params: theta_1, theta_2, projector_onehot, sample_cond_prob
    unravel_vmap = jax.vmap(unravel, in_axes=(0,), out_axes=0)
    adaptive_vqe = partial(adaptive_vqe_factory,
                            n=n_bits,
                            synd=synd,
                            corr=corr,
                            hamiltonian=hamiltonian,
                            ctype=ctype,
                            htype=htype)
    adaptive_vqe_vmap = jax.vmap(adaptive_vqe, in_axes=(0, 0, 0, 0), out_axes=0)
    # params: randkey, theta_1
    sample = partial(sample_factory,
                        generator=synd,
                        n_bits=n_bits)
    sample_vmap = jax.vmap(sample, in_axes=(0, 0), out_axes=(0, 0))
    # ---------------------- 2. Initialize variables and tensors ----------------
    batch_size = theta_1_batched.shape[0]
    theta1_num = theta_1_batched.shape[1]

    # build parameter-shifted theta_1
    theta_1_full = jnp.tile( theta_1_batched[:, None, None, :], (1, sample_round, 1)).astype(ftype) # (batch_size, sample_round, θ₁_num)

    # ------------
    def _energy(theta_1_shifted, gamma, randkey):
        """
                    Args:
                        theta_1_shifted      : shape (batch, sample_round, θ₁_num)
                        gamma                : shape (batch, γ_num)
                        randkey              : PRNGKey for sampling, shape (batch, sample_round, 2)
                    """
        # Sample projectors and conditional probabilities
        theta_1_reshaped = jnp.reshape(theta_1_shifted,
                                       (-1, theta1_num))  # (batch_size * sample_round, θ₁_num)
        gamma_reshaped = jnp.reshape(gamma,
                                     (batch_size, gamma_batched.shape[-1]))  # (batch_size, γ_num)
        gamma_converted = unravel_vmap(gamma_reshaped)  # convert to params dict for model
        randkey_reshaped = jnp.reshape(randkey, (-1, 2))  # (batch_size, sample_round, 2)

        projector, cond_prob = sample_vmap(randkey_reshaped, theta_1_reshaped)  # shape (batch_size * sample_round, n_bits), shape (batch_size * sample_round * θ₁_num,)
        measure_result = 2 * (projector - 0.5)
        # one-hot encoding of projectors
        proj_onehot = jnn.one_hot(projector, 3)  # shape (batch_size * sample_round * theta1_num, n_bits, 3)
        forward_pass_vmap = jax.vmap(forward_pass, in_axes=(None, 0, 0), out_axes=0)
        theta_2 = forward_pass_vmap(
            model,
            measure_result.reshape(batch_size, -1, n_bits),  # shape (batch_size, sample_round, n_bits)
            gamma_converted
        )  # shape (batch_size, sample_round * θ₁_num, θ₂_num)
        # params: theta_1, theta_2, projector_onehot, sample_cond_prob
        energy = adaptive_vqe_vmap(
            theta_1_reshaped,
            theta_2.reshape(batch_size * sample_round, -1),
            proj_onehot,
            cond_prob) # shape (batch_size * sample_round,)

        # Reshape back to (batch_size, sample_round, θ₁_num)
        energy = jnp.reshape(energy, (batch_size, sample_round))
        return energy

    root_key, batch_keys = make_batch_keys(input_key, batch_size * sample_round)
    energy_batched_sample = _energy(theta_1_full, gamma_batched, batch_keys) # (batch_size, sample_round)

    energy_batched = jnp.mean(energy_batched_sample, axis=1)  # (batch,)
    min_energy = jnp.min(energy_batched, axis=0)
    mean_energy = jnp.mean(energy_batched, axis=0)
    return min_energy, mean_energy