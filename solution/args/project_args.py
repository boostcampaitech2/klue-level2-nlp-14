from typing import List
from dataclasses import dataclass, field


@dataclass
class ProjectArguments:
    task: str = field(
        default="klue_re", metadata={"help": "Task name. This kwarg is used by `TASK_INFOS_MAP` and `TASK_METRIC_MAP` to get task-specific information."},
    )
    wandb_project: str = field(
        default="klue_re", metadata={"help": "weight and biases project name."},
    )
    save_model_dir: str = field(
        default="best", metadata={"help": "Directory where the trained model is stored."},
    )
    checkpoint: str = field(
        default=None, metadata={"help": "Checkpoint with models to be used for inference."},
    )
    do_analysis: bool = field(
        default=False, metadata={"help": "Whether to use analysis mode or not."},
    )
    infer_pipeline_name: str = field(
        default="basic", metadata={"help": "Inference pipeline name. You can find this object on `solution/utils/inference.py`"},
    )
