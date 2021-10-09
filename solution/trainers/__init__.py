from transformers import Trainer
from .sampler import (
    BalancedSamplerTrainer,
    DefaultTrainer,
    XLMTrainer,
)

TRAINER_MAP = {
    "default": DefaultTrainer,
    "balanced": BalancedSamplerTrainer,
    "xlm": XLMTrainer,
}
