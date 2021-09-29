import argparse
from distutils.util import strtobool
from typing import List, Optional, Dict, Any
from transformers.optimization import (
    Adafactor,
    get_cosine_schedule_with_warmup,
    get_cosine_with_hard_restarts_schedule_with_warmup,
    get_linear_schedule_with_warmup,
    get_polynomial_decay_schedule_with_warmup,
)

# https://melkia.dev/ko/questions/15008758?page=3

def add_general_args(parser: argparse.ArgumentParser, root_dir: str) -> argparse.ArgumentParser:

    # update this and the import above to support new schedulers from transformers.optimization
    # arg_to_scheduler = {
    #     "linear": get_linear_schedule_with_warmup,
    #     "cosine": get_cosine_schedule_with_warmup,
    #     "cosine_w_restarts": get_cosine_with_hard_restarts_schedule_with_warmup,
    #     "polynomial": get_polynomial_decay_schedule_with_warmup,
    # }
    # arg_to_scheduler_choices = sorted(arg_to_scheduler.keys())
    # arg_to_scheduler_metavar = "{" + ", ".join(arg_to_scheduler_choices) + "}"

    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--gpus",
        default=None,
        nargs="+",
        type=int,
        help="Select specific GPU allocated for this, it is by default [] meaning none",
    )
    parser.add_argument(
        "--fp16",
        type=lambda x: bool(strtobool(x)),
        help="Whether to use 16-bit (mixed) precision instead of 32-bit",
    )
    # parser.add_argument(
    #     "--num_sanity_val_steps",
    #     type=int,
    #     default=2,
    #     help="Sanity check validation steps (default 2 steps)",
    # )
    parser.add_argument("--max_grad_norm", dest="gradient_clip_val", default=1.0, type=float, help="Max gradient norm")
    # parser.add_argument(
    #     "--gradient_accumulation_steps",
    #     dest="accumulate_grad_batches",
    #     type=int,
    #     default=1,
    #     help="Number of updates steps to accumulate before performing a backward/update pass.",
    # )
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    parser.add_argument("--metric_key", type=str, default="loss", help="The name of monitoring metric")
    # parser.add_argument(
    #     "--patience",
    #     default=5,
    #     type=int,
    #     help="The number of validation epochs with no improvement after which training will be stopped.",
    # )
    # parser.add_argument(
    #     "--early_stopping_mode",
    #     choices=["min", "max"],
    #     default="max",
    #     type=str,
    #     help="In min mode, training will stop when the quantity monitored has stopped decreasing; in max mode it will stop when the quantity monitored has stopped increasing;",
    # )

    # Data Args
    # parser.add_argument("--data_dir", default=None, type=str, help="The input data dir", required=True)
    parser.add_argument("--num_workers", default=4, type=int, help="kwarg passed to DataLoader")
    parser.add_argument("--per_device_train_batch_size", default=32, type=int)
    parser.add_argument("--per_device_eval_batch_size", default=32, type=int)
    parser.add_argument(
        "--max_seq_length",
        default=128,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
        "than this will be truncated, sequences shorter will be padded.",
    )


    # Model
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        required=True,
        help="model identifier from huggingface.co/models",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        type=str,
        help="Path to pretrained model",
    )
    parser.add_argument(
        "--config_name", default="", type=str, help="Pretrained config name or path if not the same as model_name"
    )
    parser.add_argument(
        "--tokenizer_name",
        default=None,
        type=str,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--cache_dir",
        default="",
        type=str,
        help="Where do you want to store the pre-trained models downloaded from s3",
    )
    # parser.add_argument(
    #     "--encoder_layerdrop",
    #     type=float,
    #     help="Encoder layer dropout probability (Optional). Goes into model.config",
    # )
    # parser.add_argument(
    #     "--decoder_layerdrop",
    #     type=float,
    #     help="Decoder layer dropout probability (Optional). Goes into model.config",
    # )
    # parser.add_argument(
    #     "--dropout",
    #     type=float,
    #     help="Dropout probability (Optional). Goes into model.config",
    # )
    # parser.add_argument(
    #     "--attention_dropout",
    #     type=float,
    #     help="Attention dropout probability (Optional). Goes into model.config",
    # )
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    # parser.add_argument(
    #     "--lr_scheduler",
    #     default="linear",
    #     choices=arg_to_scheduler_choices,
    #     metavar=arg_to_scheduler_metavar,
    #     type=str,
    #     help="Learning rate scheduler",
    # )
    parser.add_argument("--num_train_epochs", default=4, type=int, help="max_epochs")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--warmup_steps", default=None, type=int, help="Linear warmup over warmup_steps.")
    parser.add_argument("--warmup_ratio", default=None, type=float, help="Linear warmup over warmup_step ratio.")

    parser.add_argument("--lr_scheduler_type", default="linear", type=str)

    # Custom
    parser.add_argument("--dataset_ver", default=None, type=str, help="dataset version")
    parser.add_argument("--report_to", default="none", type=str, help="wandb report or not 'none' vs. 'wandb'")
    parser.add_argument("--wandb_project", default=None, type=str, help="wandb project name")
    parser.add_argument("--wandb_run_name", default=None, type=str, help="wandb run_name name")
    parser.add_argument("--do_train", default=False,  help="do train", type=lambda x: bool(strtobool(x)))
    parser.add_argument("--do_eval",  default=False, help="do eval", type=lambda x: bool(strtobool(x)))
    parser.add_argument("--do_predict", default=False, help="do predict", type=lambda x: bool(strtobool(x)))

    return parser