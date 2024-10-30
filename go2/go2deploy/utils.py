import torch
import numpy as np
from typing import Union

ArrayLike = Union[np.ndarray, torch.Tensor]


def lerp(a, b, alpha):
    return a * (1 - alpha) + alpha * b


def normalize(x) -> ArrayLike:
    if isinstance(x, torch.Tensor):
        return x / torch.norm(x, dim=-1, keepdim=True)
    elif isinstance(x, np.ndarray):
        return x / np.linalg.norm(x, axis=-1, keepdims=True)