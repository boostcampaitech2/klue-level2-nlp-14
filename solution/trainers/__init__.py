from transformers import Trainer
from .sampler import (
    BalancedSamplerTrainer,
)
from .default import (
    DefaultTrainer,
)


TRAINER_MAP = {
    "default": DefaultTrainer,
    "balanced": BalancedSamplerTrainer,
}