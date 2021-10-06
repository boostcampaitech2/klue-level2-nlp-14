from transformers import Trainer
from .sampler import (
    BalancedSamplerTrainer,
    CustomTrainer
)

TRAINER_MAP = {
    "default": Trainer,
    "custom": CustomTrainer,
    "balanced": BalancedSamplerTrainer,
}