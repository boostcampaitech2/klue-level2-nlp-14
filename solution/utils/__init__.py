import os
import random
import numpy as np
import torch
import pickle

from .file_utils import TASK_INFOS_MAP, CONFIG_FILE_NAME, PYTORCH_MODEL_NAME, IDX2LABEL, LABEL2IDX
from .metrics import TASK_METRIC_MAP
from .inference import INFERENCE_PIPELINE
from .utils import softmax, set_seeds
from .loss import (
    CrossEntropyLoss,
    DiceLoss, 
    FocalLoss, 
    CrossEntropyClassWeight,
)

LOSS_MAP = {
    "default": CrossEntropyLoss,
    "focal": FocalLoss,
    "dice": DiceLoss,
    "weight": CrossEntropyClassWeight,
}