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
from flax.traverse_util import flatten_dict, unflatten_dict
import optax
tc.set_backend("jax")

def get_prob(n: int,
             generator: Callable[[int, jnp.ndarray], tc.Circuit],
             theta_1: jnp.ndarray,
             projector_onehot: jnp.ndarray,
             prob_idx: int,
             ctype: jnp.dtype = jnp.complex64) -> jnp.ndarray:
    """
    n_bits: number of system qubits
    generator: function that builds the base circuit
    theta_1: angles for circuit, no batching
    projector_onehot: shape (n_bits, 3) one-hot selectors for each qubit/projector
    prob_idx: which of the added ancilla qubits to measure
    """
    # cast inputs
    theta_1 = theta_1.astype(ctype)
    projector_onehot = projector_onehot.astype(ctype)

    # build the base circuit
    circuit = generator(n, theta_1)

    # predefine the three projectors: |0><0|, |1><1|, I
    projector_set = jnp.array([
        [[1., 0.],
         [0., 0.]],
        [[0., 0.],
         [0., 1.]],
        [[1., 0.],
         [0., 1.]]
    ], dtype=ctype)  # shape (3,2,2)

    # body of the loop: apply one projector per system qubit
    def body(idx, circ):
        # select the one-hot for qubit idx: shape (3,)
        sel = projector_onehot[idx]  # (3,)
        # compute projector unitary for this qubit
        # → sum over projector_set * one-hot scalar
        #    projector_set[0]*sel[...,0] + projector_set[1]*sel[...,1] + ...
        # we use broadcasting: (3,)[:,None,None] * (3,2,2) -> (3,2,2)
        # then sum over the projector axis:
        proj = jnp.sum(sel[:, None, None] * projector_set, axis=0)
        # then insert it on the ancilla line n_bits+idx
        return circ.any(n + idx, unitary=proj)

    # loop across all system qubits
    circuit = jax.lax.fori_loop(0, n, body, circuit)

    # finally take the expectation on the chosen ancilla (n_bits+prob_idx)
    prob = circuit.expectation_ps(z=[n + prob_idx])
    return prob


#batched_get_prob = jax.jit(jax.vmap(get_prob, in_axes=(None, None, 0, 0, None), out_axes=0), static_argnums=(0, 1, 4))

def sample_factory(randkey: jax.random.PRNGKey,
           generator: Callable[[int, jnp.ndarray], tc.Circuit],
           n_bits: int,
           theta_1: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    key: PRNGKey for sampling
    n_bits: number of system qubits
    batch_size: number of shots in parallel
    theta_1_batched: shape (batch_size,) angles for U1
    returns: (projector_batched, cond_prob_batched)
      - projector_batched: (n_bits) sampled projector indices {0,1,2}
      - cond_prob_batched: () the joint probability of the recorded projectors
    """
    # init buffers
    bit_prob= jnp.zeros((n_bits, 2), dtype=jnp.float32)
    projector = 2 * jnp.ones((n_bits,), dtype=jnp.float32)  # start “identity”(2)
    cond_prob = jnp.ones((1,), dtype=jnp.float32)

    def body(carry, idx):
        key, proj, prob, cond = carry
        key, subkey = jax.random.split(key)

        # — if idx>0, sample the (idx-1) projector based on previous probs —
        def sample_prev(state):
            proj_prev, cond_prev = state
            logits = jnp.log(prob[idx - 1, :])  # (2,)
            draw = jax.random.categorical(subkey, logits)  # ()
            proj_prev = proj_prev.at[idx - 1].set(draw.astype(jnp.float32))

            # update conditional probability
            oh = jax.nn.one_hot(draw, 2, dtype=jnp.float32)  # (2,)
            cond_prev = jnp.sum(cond_prev * (prob[idx - 1, :] * oh),
                                axis=-1)
            return proj_prev, cond_prev

        proj, cond = jax.lax.cond(idx > 0, sample_prev, lambda s: s, operand=(proj, cond))

        # — now force the current idx to projector=0 and compute its prob —
        # proj = proj.at[idx].set(0.)

        # make one-hot over {0,1,2} for all qubits so far
        proj_onehot = jax.nn.one_hot(proj.astype(jnp.int32), 3, dtype=jnp.float32)
        # compute unconditioned P(0) on ancilla n_bits+idx
        expect_z_raw = get_prob(n_bits, generator, theta_1, proj_onehot, idx)  # ()
        expect_z = expect_z_raw.astype(jnp.float32) / cond
        p0 = jnp.clip((1 + expect_z) / 2, 0.0, 1.0)
        # store P(0) and P(1)=1−P(0)
        prob = prob.at[idx, 0].set(p0)
        prob = prob.at[idx, 1].set(1 - p0)

        return (key, proj, prob, cond), None

    # run through all system qubits
    (key, projector, prob, cond_prob), _ = \
        jax.lax.scan(body,
                     (randkey, projector, bit_prob, cond_prob),
                     jnp.arange(n_bits))

    # finally sample the last projector (idx = m-1)
    key, subkey = jax.random.split(key)
    last_logits = jnp.log(prob[-1, :])
    last_draw = jax.random.categorical(subkey, last_logits)
    projector = projector.at[n_bits - 1].set(last_draw.astype(jnp.float32))

    # update final conditional probability
    oh_last = jax.nn.one_hot(last_draw, 2, dtype=jnp.float32)
    cond_prob = jnp.sum(cond_prob * (prob[-1, :] * oh_last), axis=-1)

    return projector, cond_prob