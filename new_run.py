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
    mark_entity_spans as _mark_entity_spans,
    convert_example_to_features as _convert_example_to_features,
)
from solution.utils import (
    softmax,
    set_seeds,
    TASK_METRIC_MAP,
    TASK_INFOS_MAP,
    CONFIG_FILE_NAME,
    PYTORCH_MODEL_NAME,
)
import solution.models as models


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
    data_collator = collate_cls(tokenizer)
    
    # Preprocess and tokenizing
    mark_entity_spans = partial(_mark_entity_spans, **task_infos.markers)
    convert_example_to_features = partial(
        _convert_example_to_features,
        tokenizer=tokenizer,
        **task_infos.markers,
    )
    
    examples = dataset.map(mark_entity_spans)
    tokenized_datasets = examples.map(convert_example_to_features)
    
    # Get model
    def model_init():
        config = AutoConfig.from_pretrained(
            model_args.model_name_or_path,
            num_labels=task_infos.num_labels,
            cache_dir=model_args.model_cache_dir,
            id2label=task_infos.id2label,
            label2id=task_infos.label2id,
        )
        model_cls = getattr(models, model_args.architectures,
                            AutoModelForSequenceClassification)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_args.model_name_or_path,
            config=config,
            cache_dir=model_args.model_cache_dir,
        )
        if model.config.vocab_size < len(tokenizer):
            print("resize...")
            model.resize_token_embeddings(len(tokenizer))
        return model
            
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
        
    # remove unused feature names
    features_name = list(tokenized_datasets["train"].features.keys())
    features_name.pop(features_name.index("input_ids"))
    features_name.pop(features_name.index("label"))
    tokenized_datasets = tokenized_datasets.remove_columns(features_name)
    
    train_dataset = tokenized_datasets["train"]
    eval_dataset = None
    if training_args.do_eval:
        try:
            eval_dataset = tokenized_datasets["valid"]
        except KeyError:
            print("Dataset Version Error")
            return None
    
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
        examples = test_dataset.map(mark_entity_spans)
        tokenized_test_datasets = examples.map(convert_example_to_features)

        features_name = list(tokenized_test_datasets.features.keys())
        features_name.pop(features_name.index("input_ids"))
        # features_name.pop(features_name.index("label"))
        tokenized_test_datasets = tokenized_test_datasets.remove_columns(features_name)
        
        logits = trainer.predict(tokenized_test_datasets)[0]
        probs = softmax(logits).tolist()
        result = np.argmax(logits, axis=-1).tolist()
        pred_answer = [task_infos.id2label[v] for v in result]
        
        ## make csv file with predicted answer
        #########################################################
        # 아래 directory와 columns의 형태는 지켜주시기 바랍니다.
        output = pd.DataFrame(
            {
                'id':test_id,
                'pred_label':pred_answer,
                'probs':probs,
            }
        )
        submir_dir = project_args.submit_dir
        run_name = training_args.run_name
        output.to_csv(f'./{submir_dir}/submission_{run_name}.csv', index=False)
        #### 필수!! ##############################################
    print('---- Finish! ----')
        

if __name__ == "__main__":
    main()