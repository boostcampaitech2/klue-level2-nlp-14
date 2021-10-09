from transformers import TrainingArguments
from dataclasses import dataclass, field


@dataclass
class NewTrainingArguments(TrainingArguments):
    """ Arguments related to training.
    Attributes:
        trainer_class:  The name of the mapping with the trainer.
    """
    trainer_class: str = field(
        default="default", metadata={"help": ""},
    ),
    loss : str = field(
        default="default", metadata={"help": ""},
    )