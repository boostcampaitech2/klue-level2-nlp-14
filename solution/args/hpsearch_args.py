from typing import List, Union
from dataclasses import dataclass, field

from .argparse import lambda_field


@dataclass
class HPSearchArguments:
    smoke_test: bool = field(
        default=True, metadata={"help": "Finish quickly for testing"},
    )
    ray_address: str = field(
        default=None, metadata={"help": "Address to use for Ray. "
                                      "Use 'auto' for cluster. "      
                                      "Defaults to None for local."},
    )
    server_address: str = field(
        default=None, metadata={"help": "The address of server to connect to "
                                        "if using Ray Client."},
    )
    method: str = field(
        default="ASHA", metadata={"help": ""},
    )
    objective_metric: str = field(
        default="eval_auprc", metadata={"help": ""},
    )
    hp_per_device_train_batch_size: List[int] = lambda_field(
        default=[8, 16, 32], metadata={"help": ""},
    )
    hp_per_device_eval_batch_size: Union[List[int]] = lambda_field(
        default=32, metadata={"help": ""},
    )
    hp_learning_rate: List[float] = lambda_field(
        default=[1e-5, 2e-5, 3e-5, 5e-5], metadata={"help": ""},
    )
    hp_warmup_ratio: List[float] = lambda_field(
        default=[0., 0.1, 0.2, 0.6], metadata={"help": ""},
    )
    hp_weight_decay: List[float] = lambda_field(
        default=[0.0, 0.01], metadata={"help": ""},
    )
    hp_num_train_epochs: Union[List[float]] = lambda_field(
        default=1, metadata={"help": ""},
    )