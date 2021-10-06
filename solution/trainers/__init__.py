from transformers import Trainer
from .sampler import (
    BalancedSamplerTrainer,
    DefaultTrainer,
)

TRAINER_MAP = {
    "default": DefaultTrainer,
    "balanced": BalancedSamplerTrainer,
}
