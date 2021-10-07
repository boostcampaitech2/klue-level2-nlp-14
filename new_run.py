import os
import sys

from functools import partial
from typing import Tuple, List, Any, Dict

import numpy as np
import pandas as pd

import torch
import torch.nn as nn

from datasets import load_dataset, concatenate_datasets

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
    NewTrainingArguments,
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
from solution.trainers import (
    TRAINER_MAP,
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


# debug for cuda
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def main():
    parser = HfArgumentParser(
        (DataArguments,
         NewTrainingArguments,
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
    prep_pipeline = PREPROCESSING_PIPELINE[data_args.prep_pipeline_name]
    data_collator = collate_cls(tokenizer)
    
    # Preprocess and tokenizing
    tokenized_datasets = prep_pipeline(dataset,
                                       tokenizer,
                                       task_infos,)
    
    # =============================================================
    # UnitTest input
    # print(tokenized_datasets["train"][0])
    # return None
    # =============================================================
    
    # Get model
    _model_init = MODEL_INIT_FUNC[model_args.model_init]
    model_init = partial(_model_init, 
                         model_args=model_args,
                         task_infos=task_infos,
                         tokenizer=tokenizer,)
    
    # =============================================================
    # UnitTest RECENT
    # model = model_init()
    # output = model(torch.LongTensor(2, 10).random_(10000),
    #                head_ids=torch.LongTensor([0,8]),
    #                labels=torch.LongTensor(
    #                    [[0,0,4,0,0,2,0,0,0,0,0,0,0,],
    #                     [0,8,0,0,0,0,0,0,0,0,1,0,0,]]))
    # return output
    # =============================================================
    
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
        
    if project_args == "tapt":
        test_dataset = load_dataset(data_args.name, 
                                    script_version=data_args.revision, 
                                    cache_dir=data_args.data_cache_dir,
                                    split="test",)
        test_dataset = prep_pipeline(test_dataset,
                                     tokenizer,
                                     task_infos,
                                     mode="train",)
        train_dataset = concatenate_datasets([train_dataset, test_dataset])
        eval_dataset = None
        
    # =============================================================
    # Smoke test
    # train_dataset = train_dataset.select([i for i in range(1000)])
    # if eval_dataset is not None:
    #     eval_dataset = eval_dataset.select([i for i in range(500)])
    # =============================================================

<<<<<<< HEAD
    trainer = Trainer(
        model=model_init(),
=======
    trainer_class = TRAINER_MAP[training_args.trainer_class]
    trainer = trainer_class(
>>>>>>> edc6c0fcc60fecf18aac5318772a4654d2cdf616
        args=training_args,
        # model_init=model_init,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    
    # Training
    if training_args.do_train:
        trainer.train()
        checkpoint = project_args.save_model_dir
        trainer.model.save_pretrained(checkpoint)
        tokenizer.save_pretrained(checkpoint)
    
    if training_args.do_predict and not project_args.task == "tapt":
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
        tokenized_test_datasets = prep_pipeline(test_dataset,
                                                tokenizer,
                                                task_infos,
                                                mode="test",)
        
        infer_pipeline = INFERENCE_PIPELINE[project_args.infer_pipeline_name]
        probs, pred_answer = infer_pipeline(tokenized_test_datasets,
                                            trainer=trainer,
                                            task_infos=task_infos,
                                            training_args=training_args,)
        output = pd.DataFrame(
            {
                'id':test_id,
                'pred_label':pred_answer,
                'probs':probs,
            }
        )
        submir_dir = training_args.output_dir
        run_name = training_args.run_name
        output.to_csv(f'{submir_dir}/submission_{run_name}.csv', index=False)
    print('---- Finish! ----')
        

if __name__ == "__main__":
    main()