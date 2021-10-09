from transformers import TrainingArguments
from dataclasses import dataclass, field


@dataclass
class NewTrainingArguments(TrainingArguments):
    trainer_class: str = field(
        default="default", metadata={"help": ""},
    ),
    loss : str = field(
        default="default", metadata={"help": ""},
    )