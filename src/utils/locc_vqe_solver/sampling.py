from collections.abc import Callable
from jax import config
# Must happen before any JAX imports
config.update("jax_enable_x64", True)
import tensorcircuit as tc
import jax
import jax.numpy as jnp
import jax.nn as jnn
tc.set_backend("jax")

def get_prob(n: int,
             generator: Callable[[int, jax.Array], tc.Circuit],
             theta_1: jax.Array,
             projector_onehot: jax.Array,
             prob_idx: int,
             ctype: jnp.dtype = jnp.complex128) -> jax.Array:
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

    for idx in range(n):
        # select the one-hot for qubit idx: shape (3,)
        sel = projector_onehot[idx]  # (3,)
        # compute projector unitary for this qubit
        proj = jnp.sum(sel[:, None, None] * projector_set, axis=0)
        # then insert it on the ancilla line n_bits+idx
        circuit.any(n + idx, unitary=proj)

    # finally take the expectation on the chosen ancilla (n_bits+prob_idx)
    prob = circuit.expectation_ps(z=[n + prob_idx])
    return jnp.real(prob)


#batched_get_prob = jax.jit(jax.vmap(get_prob, in_axes=(None, None, 0, 0, None), out_axes=0), static_argnums=(0, 1, 4))

def sample_factory(
    randkey: jax.random.PRNGKey,
    theta_1: jax.Array,
    generator: Callable[[int, jax.Array], "tc.Circuit"],
    n_bits: int,
    ftype: jnp.dtype = jnp.float64
) -> tuple[jax.Array, jax.Array]:
    # initialise projector state: 2 denotes identity (no measurement yet)
    projector_init = 2 * jnp.ones((n_bits,), dtype=ftype)
    cond_init = jnp.ones((), dtype=ftype)

    def body(carry, idx):
        key, proj, cond = carry
        key, subkey = jax.random.split(key)

        proj_tmp = proj.at[idx].set(2.)
        proj_onehot = jnn.one_hot(proj_tmp.astype(jnp.int32), 3, dtype=ftype)

        # Instead of using vmap, compute each expectation explicitly
        expect_vec = jnp.zeros(n_bits, dtype=ftype)
        for j in range(n_bits):
            expect_vec = expect_vec.at[j].set(
                get_prob(n_bits, generator, theta_1, proj_onehot, j)
            )

        expect_z_raw = expect_vec[idx]
        expect_z = expect_z_raw.astype(ftype) / cond
        p0 = jnp.clip((1.0 + expect_z) / 2.0, 0.0, 1.0)
        probs = jnp.stack([p0, 1.0 - p0])

        draw = jax.random.categorical(subkey, jnp.log(probs))
        proj = proj.at[idx].set(draw.astype(ftype))
        cond = cond * probs[draw]

        return (key, proj, cond), None


    # scan over all qubits
    (_, projector, cond_prob), _ = jax.lax.scan(
        body, (randkey, projector_init, cond_init), jnp.arange(n_bits)
    )

    return projector, cond_prob
