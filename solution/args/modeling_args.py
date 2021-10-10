from typing import List
from dataclasses import dataclass, field


@dataclass
class ModelingArguments:
    model_name_or_path: str = field(
        default="klue/roberta-large", metadata={"help": "Model identifier from huggingface.co/models"}
    )
    architectures: str = field(
        default="AutoModelForSequenceClassification",
        metadata={"help": "Model architectures. You can find this object on `solution/models`"},
    )
    model_cache_dir: str = field(
        default="cache", metadata={"help": "Model cache directory path"},
    )
    model_init: str = field(
        default="basic", metadata={"help": "Which function to use to initialize the model?"},
    )
    dense_type: str = field(
        default="Linear", metadata={"help": "Dense type for recent multiple head. ['Linear', 'LSTM']"},
    )
    act_type: str = field(
        default="tanh", metadata={"help": "Activation type for recent multiple head. ['tanh', 'relu']"},
    )
