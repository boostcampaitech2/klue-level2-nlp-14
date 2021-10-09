from transformers import RobertaForSequenceClassification
import math

import torch
import torch.utils.checkpoint
from packaging import version
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers.models.roberta import RobertaModel
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions, SequenceClassifierOutput
from transformers.file_utils import ModelOutput
from collections import OrderedDict, UserDict

from transformers.modeling_utils import PreTrainedModel
