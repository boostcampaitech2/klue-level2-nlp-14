from typing import List
from dataclasses import dataclass, field


@dataclass
class ModelingArguments:
    """ Arguments related to modeling.
    Attributes:
        model_name_or_path:     The name or path of the model.
        architectures:          The architectures of the model.
        model_cache_dir:        The directory where the model will be cached.
    """
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