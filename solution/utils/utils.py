import os
import random
import numpy as np
import torch


def softmax(arr: np.ndarray, axis: int = -1):
    """ Applies the Softmax function to an n-dimensional input Tensor rescaling them """
    c = arr.max(axis=axis, keepdims=True)
    s = arr - c
    nominator = np.exp(s)
    denominator = nominator.sum(axis=axis, keepdims=True)
    probs = nominator / denominator
    return probs


def set_seeds(seed=42):
    """ A function that fixes a random seed for reproducibility """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False # for faster training, but not deterministic
