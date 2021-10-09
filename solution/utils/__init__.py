import os
import random
import numpy as np
import torch
import pickle

from .file_utils import (
    TASK_INFOS_MAP,
    CONFIG_FILE_NAME,
    PYTORCH_MODEL_NAME,
    RELATION_CLASS,
)
from .metrics import TASK_METRIC_MAP, get_confusion_matrix
from .inference import INFERENCE_PIPELINE
from .utils import softmax, set_seeds
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


def label_to_num(label):
    num_label = []
    with open('dict_label_to_num.pkl', 'rb') as f:
        dict_label_to_num = pickle.load(f)
    for v in label:
        num_label.append(dict_label_to_num[v])

    return num_label
