from typing import List, Union
from dataclasses import dataclass, field

from .argparse import lambda_field


@dataclass
class HPSearchArguments:
    """ Arguments related to hyperparameter search.
    Attributes:
        smoke_test:             Whether it's a smoke test or not.
        ray_address:            The address to use for ray.
        server_address:         The address of server to connect to.
        method:                 Hyperparameter search method.
        backend:                Hyperparameter search library.
        num_samples:            The number of random search trials.
        save_ckpt:              Whether to save the checkpoint.
        objective_metric:       The objective metric of hyperparameter search.
        hp_per_device_train_batch_size: The batch size per GPU/TPU core/CPU for training.
        hp_per_device_eval_batch_size:  The batch size per GPU/TPU core/CPU for evaluation.
        hp_learning_rate:       The initial learning rate for optimizer.
        hp_warmup_ratio:        Ratio of total training steps used for a linear warmup from 0 to learning_rate.
        hp_num_train_epochs:    Total number of training epochs to perform.
        hp_weight_decay:        The weight decay to apply to layers.
    """
    smoke_test: bool = field(
        default=False, metadata={"help": "Finish quickly for testing"},
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
    backend: str = field(
        default="optuna", metadata={"help": "Hyperparameter search libarary. [optuna, ray]"},
    )
    num_samples: int = field(
        default=2, metadata={"help": "number of random search trials"},
    )
    save_ckpt: bool = field(
        default=False, metadata={"help": "Save checkpoint"},
    )
    objective_metric: str = field(
        default="eval_auprc", metadata={"help": "[eval_auprc, eval_f1,eval_loss]"},
    )
    hp_per_device_train_batch_size: List[int] = lambda_field(
        default=[8, 16, 32], metadata={"help": ""},
    )
    hp_per_device_eval_batch_size: Union[List[int]] = lambda_field(
        default=32, metadata={"help": ""},
    )
    hp_learning_rate: List[float] = lambda_field(
        default=[1e-5, 2e-5, 3e-5, 5e-5], metadata={"help": "Input choices categorically e.g. [0.0, 0.1, 0.3]"},
    )
    hp_warmup_ratio: List[float] = lambda_field(
        default=[0., 0.1, 0.2, 0.6], metadata={"help": "Input choices categorically e.g. [0.0, 0.1, 0.3]"},
    )
    hp_num_train_epochs: List[float] = lambda_field(
        default=[3, 4, 5, 10], metadata={"help": "Input choices categorically e.g. [0.0, 0.1, 0.3]"},
    )
    hp_weight_decay: List[float] = lambda_field(
        default=[0.0, 0.01], metadata={"help": "Input choices categorically e.g. [0.0, 0.1, 0.3]"},
    )