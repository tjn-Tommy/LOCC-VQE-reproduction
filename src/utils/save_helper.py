import numpy as np

def to_py(obj):
    if isinstance(obj, np.ndarray) or hasattr(obj, "tolist"):
        return obj.tolist()
    if hasattr(obj, "__float__"):
        return float(obj)
    if hasattr(obj, "__int__"):
        return int(obj)
    if isinstance(obj, dict):
        return {k: to_py(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_py(v) for v in obj]
    return obj