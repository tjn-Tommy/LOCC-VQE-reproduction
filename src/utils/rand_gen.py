
import jax
from typing import Any


def make_batch_keys(input_key: Any,
                    batch_size: int) -> tuple[Any, Any]:
    """
    Create `batch_size` independent PRNG keys starting from a single seed.

    Args:
        input_key: jax.random.PRNGKey, the initial key to split from.
        batch_size: int, the number of keys to generate.
    Returns:
        tuple: (root_key, batch_keys)
            - root_key: jax.random.PRNGKey, the first key for the main computation.
            - batch_keys: jax.Array, an array of PRNG keys for each batch.
    """
    # split into batch_size new keys
    batch_keys = jax.random.split(input_key, batch_size + 1)
    return batch_keys[0], batch_keys[1:]