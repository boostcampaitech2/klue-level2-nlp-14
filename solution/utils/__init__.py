import os
import random
import numpy as np
import torch

from .file_utils import TASK_INFOS_MAP, CONFIG_FILE_NAME, PYTORCH_MODEL_NAME
from .metrics import TASK_METRIC_MAP
from .loss import (
    DiceLoss, 
    FocalLoss, 
    CrossEntropyClassWeight,
)

LOSS_MAP = {
    "default": DiceLoss,
    "focal": FocalLoss,
    "dice": DiceLoss,
    "weight": CrossEntropyClassWeight,
}


def softmax(arr: np.ndarray, axis: int = -1):
    c = arr.max(axis=axis, keepdims=True)
    s = arr - c
    nominator = np.exp(s)
    denominator = nominator.sum(axis=axis, keepdims=True)
    probs = nominator / denominator
    return probs


def set_seeds(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False # for faster training, but not deterministic
    
    
