import os
import numpy as np
import torch


def softmax(arr: np.ndarray, axis: int = -1):
    c = arr.max(axis=axis, keepdims=True)
    s = arr - c
    nominator = np.exp(s)
    denominator = nominator.sum(axis=axis, keepdims=True)
    probs = nominator / denominator
    return probs
