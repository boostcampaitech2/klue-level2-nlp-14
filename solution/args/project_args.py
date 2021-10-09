from typing import List
from dataclasses import dataclass, field


@dataclass
class ProjectArguments:
    """ Arguments related to project.
    Attributes:
        task:               The name of the project task.
        wandb_project:      The name of the wandb project.
        save_model_dir:     The path of the directory to save the model.
        submit_dir:         The path of the directory to store the prediction results.
        checkpoint:         The path of the directory where the checkpoint is to be saved.
    """
    task: str = field(
        default="klue_re", metadata={"help": ""},
    )
    wandb_project: str = field(
        default="klue_re", metadata={"help": ""},
    )
    save_model_dir: str = field(
        default="best", metadata={"help": ""},
    )
    checkpoint: str = field(
        default=None, metadata={"help": ""},
    )
    do_analysis: bool = field(
        default=False, metadata={"help": ""},
    )
    infer_pipeline_name: str = field(
        default="basic", metadata={"help": ""},
    )
