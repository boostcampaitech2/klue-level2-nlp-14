import os
import sys

from functools import partial
from typing import Tuple, List, Any, Dict

import numpy as np
import pandas as pd

import torch
import torch.nn as nn

from datasets import load_dataset

import transformers
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForSequenceClassification,
)
from transformers import TrainingArguments, Trainer
from transformers.configuration_utils import PretrainedConfig

from solution.args import (
    HfArgumentParser,
    DataArguments,
    TrainingArguments,
    ModelingArguments,
    ProjectArguments,
)
from solution.data import (
    COLLATOR_MAP,
    PREPROCESSING_PIPELINE,
)
from solution.models import (
    MODEL_INIT_FUNC,
)
from solution.utils import (
    softmax,
    set_seeds,
    TASK_METRIC_MAP,
    TASK_INFOS_MAP,
    CONFIG_FILE_NAME,
    PYTORCH_MODEL_NAME,
    INFERENCE_PIPELINE,
)


def main():
    parser = HfArgumentParser(
        (DataArguments,
         TrainingArguments,
         ModelingArguments,
         ProjectArguments,)
    )
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # read args from json file
        args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    elif len(sys.argv) == 2 and sys.argv[1].endswith(".yaml"):
        args = parser.parse_yaml_file(yaml_file=os.path.abspath(sys.argv[1]))
    else:
        # read args from shell script or real arguments
        args = parser.parse_args_into_dataclasses()
        
    data_args, training_args, model_args, project_args = args
    
    # Set seed
    set_seeds(training_args.seed)
    
    checkpoint = project_args.checkpoint
    
    task_infos = TASK_INFOS_MAP[project_args.task]
    compute_metrics = TASK_METRIC_MAP[project_args.task]
    
    # Get training data
    dataset = load_dataset(
        data_args.name, 
        script_version=data_args.revision, 
        cache_dir=data_args.data_cache_dir,
    )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    tokenizer.add_special_tokens(
        {"additional_special_tokens": list(task_infos.markers.values())}
    )
    collate_cls = COLLATOR_MAP[data_args.collator_name]
    prep_pipeline = PREP_PIPELINE[data_args.prep_pipeline_name]
    data_collator = collate_cls(tokenizer)
    
    # Preprocess and tokenizing
    tokenized_datasets = prep_pipeline(dataset,
                                       tokenizer,
                                       task_infos,)
    
    # Get model
    _model_init = MODEL_INIT_FUNC[model_args.model_init]
    model_init = partial(_model_init, 
                         model_args=model_args, 
                         tokenizer=tokenizer,)
            
    # Set-up WANDB
    os.environ["WANDB_PROJECT"] = project_args.wandb_project

    call_wandb = True
    try:
        os.environ["WANDB_PROJECT"]

    except KeyError:
        call_wandb = False

    if call_wandb:
        import wandb
        wandb.login()
    
    # TODO datasetdict가 아닌 경우 처리
    train_dataset = tokenized_datasets["train"]
    try:
        eval_dataset = tokenized_datasets["valid"]
    except KeyError:
        eval_dataset = None

    trainer = Trainer(
        args=training_args,
        model_init=model_init,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    
    # Training
    if training_args.do_train:
        trainer.train()
        trainer.model.save_pretrained(project_args.save_model_dir)
        checkpoint = project_args.save_model_dir
    
    if training_args.do_predict:
        # Load Checkpoint
        ckpt_config_file = os.path.join(checkpoint, CONFIG_FILE_NAME)
        ckpt_model_file = os.path.join(checkpoint, PYTORCH_MODEL_NAME)
        config = PretrainedConfig.from_json_file(ckpt_config_file)
        # We load the model state dict on the CPU to avoid an OOM error.
        state_dict = torch.load(ckpt_model_file, map_location="cpu")
        # If the model is on the GPU, it still works!
        trainer._load_state_dict_in_model(state_dict)
        del state_dict
        
        # Inference
        test_dataset = load_dataset(
            data_args.name, 
            script_version=data_args.revision, 
            cache_dir=data_args.data_cache_dir,
            split="test",
        )
        test_id = test_dataset["guid"]
        tokenized_test_datasets = pipeline(test_dataset,
                                           tokenizer,
                                           task_infos,)
        tokenized_test_datasets = tokenized_test_datasets.remove_columns(["label"])
        
        infer_pipeline = INFERENCE_PIPELINE[model_args.infer_pipeline_name]
        infer_pipeline(tokenized_test_datasets,
                       task_infos,
                       training_args
                      )
    print('---- Finish! ----')
        

if __name__ == "__main__":
    main()