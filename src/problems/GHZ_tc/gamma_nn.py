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
# Use the same SimpleNet for consistency
class SimpleNet(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(features=20)(x)
        x = nn.relu(x)
        x = nn.Dense(features=8)(x)
        return x

def init_simple_net(rng: jax.random.PRNGKey,
                    n_bits: int = 8,
                    batch_size: int = 1):
    """
    Instantiate SimpleNet and init its parameters.

    Args:
      rng:           a JAX PRNGKey for parameter initialization
      n_bits: number of features in your input vector
      batch_size:     how many examples to pack into the dummy init call

    Returns:
      model:  the SimpleNet() instance
      params: a pytree of initialized weights & biases
    """
    # 1) build the model
    model = SimpleNet()

    # 2) create a dummy input of shape (batch_size, input_features)
    dummy_in = jnp.zeros((batch_size, n_bits))

    # 3) init returns a dict of collections; we pull out 'params'
    variables = model.init({'params': rng}, dummy_in)
    params = variables['params']
    # flatten the params to a 1D array
    params = jax.tree_util.tree_map(lambda x: jnp.reshape(x, (-1,)), params)
    return model, params

