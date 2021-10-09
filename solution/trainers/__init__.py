from transformers import Trainer
from .sampler import (
    BalancedSamplerTrainer,
    CustomTrainer,
    XLMTrainer,
)

TRAINER_MAP = {
    "default": Trainer,
    "custom": CustomTrainer,
    "balanced": BalancedSamplerTrainer,
    "xlm": XLMTrainer,
}