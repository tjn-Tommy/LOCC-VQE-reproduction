from collections.abc import Callable
import flax.linen as nn
import tensorcircuit as tc
import jax
import jax.numpy as jnp
from jax._src.flatten_util import ravel_pytree

tc.set_backend("jax")
# Use the same SimpleNet for consistency
class SimpleNet(nn.Module):
    @nn.compact
    def __call__(self, x):
        out = nn.Dense(features=18)(x)
        x = nn.Dense(features=96)(x)
        x = nn.relu(x)
        x = nn.Dense(features=96)(x)
        x = nn.relu(x)
        x = nn.Dense(features=18)(x)
        x = x + out  # residual connection
        x = nn.tanh(x)
        x = x * jnp.pi
        return x

def init_simple_net(rng: jax.random.PRNGKey,
                    n_bits: int,
                    ) -> jax.Array:
    """
    Instantiate SimpleNet and init its parameters.

    Args:
      rng:           a JAX PRNGKey for parameter initialization
      n_bits: number of features in your input vector

    Returns:
      model:  the SimpleNet() instance
      params: a pytree of initialized weights & biases
    """
    # 1) build the model
    model = SimpleNet()
    batch_size = 1
    # 2) create a dummy input of shape (batch_size, input_features)
    dummy_in = jnp.zeros((batch_size, n_bits))

    # 3) init returns a dict of collections; we pull out 'params'
    variables = model.init({'params': rng}, dummy_in)
    params = variables['params']
    # flatten all the parameters into a single vector
    flat_params, unravel_fn = ravel_pytree(params)
    return flat_params

def get_unravel(n_bits) -> Callable:
    """
    Instantiate SimpleNet and init its parameters.

    Args:
      n_bits: number of features in your input vector

    Returns:
      model:  the SimpleNet() instance
      params: a pytree of initialized weights & biases
    """
    # 1) build the model
    model = SimpleNet()
    batch_size = 1
    rng = jax.random.PRNGKey(42)
    # 2) create a dummy input of shape (batch_size, input_features)
    dummy_in = jnp.zeros((n_bits,))

    # 3) init returns a dict of collections; we pull out 'params'
    variables = model.init({'params': rng}, dummy_in)
    params = variables['params']
    # flatten all the parameters into a single vector
    flat_params, unravel_fn = ravel_pytree(params)
    return unravel_fn