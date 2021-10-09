from transformers import Trainer
from .sampler import (
    BalancedSamplerTrainer,
)


TRAINER_MAP = {
    "default": Trainer,
    "balanced": BalancedSamplerTrainer,
}