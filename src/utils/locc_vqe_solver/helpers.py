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
from jax.flatten_util import ravel_pytree
tc.set_backend("jax")

def convert_ndarray_to_params_single(
    model: nn.Module,
    input_size: int,
    raw_array: jnp.ndarray,
) -> dict:
    """
    Given:
      • model:        your Flax nn.Module
      • example_input: a dummy input of the same shape your real x will have
      • raw_array:     your 2-D ndarray containing exactly one param’s worth of data
    Returns:
      • a dict of the form {'Dense_0': {'kernel': raw_array}}   (or whatever your model’s leaf name is)
    """
    # 1) init to discover the tree structure
    rng = jax.random.PRNGKey(0)
    example_input = jnp.zeros((1, input_size))  # shape (1, input_size)
    init_vars = model.init(rng, example_input)
    params_tree = init_vars['params']

    # 2) flatten it so we can grab the one key path
    flat_params = flatten_dict(unfreeze(params_tree), sep='/')
    if len(flat_params) != 1:
        raise ValueError(f"Expected exactly one parameter in the model, but found {len(flat_params)}.")
    param_path, _ = next(iter(flat_params.items()))  # param_path is a tuple of strings

    # 3) build a new flat dict with that same key but your array
    new_flat = {param_path: jnp.array(raw_array)}

    # 4) unflatten back into a nested dict
    new_params_unfrozen = unflatten_dict(new_flat, sep='/')
    return new_params_unfrozen

convert_ndarray_to_params = jax.vmap(convert_ndarray_to_params_single, in_axes=(None, None, 0), out_axes=0)

def forward_pass(model: nn.Module,
                x: jnp.ndarray, #batched
                params: dict,
                 ) -> jnp.ndarray:
    # Note: We are using the 'params' variable from the outer scope.
    return model.apply({'params': params}, x)




def jacobian_wrt_params(model: nn.Module,
                             x: jnp.ndarray,
                             params: dict) -> jnp.ndarray:
    """
    Computes ∂y / ∂θ for *every* parameter θ in the model, all laid out
    in one big Jacobian array of shape (bs, y_size, total_num_params).
    """

    # 1) Flatten the pytree `params` → a single 1D vector
    flat_params, unravel_fn = ravel_pytree(params)
    #    flat_params.shape = (P,)  where P = sum of all param‐leaf sizes

    # 3) Define a function that: flat_params → model output
    def f(flat_p):
        # 3a) Reconstruct the original pytree from the flat vector
        p = unravel_fn(flat_p)
        # 3b) Run a forward pass: returns shape (bs, y_size)
        return forward_pass(model, x, p)

    # 4) Use reverse‐mode AD to get the Jacobian:
    #    f: ℝ^P → ℝ^(bs × y_size),
    #    so jacrev(f) has shape (bs, y_size, P)
    jacobian = jax.jacrev(f)(flat_params)

    return jacobian


def make_batch_keys(input_key: Any,
                    batch_size: int) -> jax.ndarray:
    """
    Create `batch_size` independent PRNG keys starting from a single seed.

    Args:
        input_key: jax.random.PRNGKey, the initial key to split from.
        batch_size: int, the number of keys to generate.
    Returns:
        tuple: (root_key, batch_keys)
            - root_key: jax.random.PRNGKey, the first key for the main computation.
            - batch_keys: jax.ndarray, an array of PRNG keys for each batch.
    """
    # split into batch_size new keys
    batch_keys = jax.random.split(input_key, batch_size + 1)
    return batch_keys[0], batch_keys[1:]