from typing import List
from dataclasses import dataclass, field


@dataclass
class ProjectArguments:
    task: str = field(
        default="klue_re", metadata={"help": ""},
    )
    wandb_project: str = field(
        default="klue_re", metadata={"help": ""},
    )
    save_model_dir: str = field(
        default="best", metadata={"help": ""},
    )
    submit_dir: str = field(
        default=None, metadata={"help": ""},
    )
    checkpoint: str = field(
        default=None, metadata={"help": ""},
    )