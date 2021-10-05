from typing import List
from dataclasses import dataclass, field


@dataclass
class ModelingArguments:
    model_name_or_path: str = field(
        default="klue/roberta-large", metadata={"help": "model identifier from huggingface.co/models"}
    )
    architectures: str = field(
        default="AutoModelForSequenceClassification",
        metadata={"help": ""},
    )
    model_cache_dir: str = field(
        default="cache", metadata={"help": ""},
    )
    model_init: str = field(
        default="basic", metadata={"help": ""},
    )
    dense_type: str = field(
        default="Linear", metadata={"help": ""},
    )
    act_type: str = field(
        default="tanh", metadata={"help": ""},
    )