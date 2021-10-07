import os
import sys
import argparse
from tqdm import tqdm

from functools import partial
from typing import Tuple, List, Any, Dict

import numpy as np
import pandas as pd

import torch
import torch.nn as nn

from sklearn.metrics import confusion_matrix
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
    PREP_PIPELINE,
    kfold_split,
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
)
import solution.models as models


def main(command_args):
    parser = HfArgumentParser(
        (DataArguments,
         NewTrainingArguments,
         ModelingArguments,
         ProjectArguments,)
    )
    if command_args.config.endswith(".json"):
        # read args from json file
        args = parser.parse_json_file(json_file=os.path.abspath(command_args.config))
    elif command_args.config.endswith(".yaml"):
        args = parser.parse_yaml_file(yaml_file=os.path.abspath(command_args.config))
    else:
        # read args from shell script or real arguments
        args = parser.parse_args_into_dataclasses()
        
    data_args, training_args, model_args, project_args = args
    
    # Set seed
    set_seeds(training_args.seed)
    
    checkpoint = project_args.checkpoint
    
    task_infos = TASK_INFOS_MAP[project_args.task]
    
    
    #del trainer
    torch.cuda.empty_cache()

    # Load & Preprocess the Dataset(for all samples of train dataset)
    train_dataset = load_dataset(
        data_args.name, 
        script_version=data_args.revision, 
        cache_dir=data_args.data_cache_dir,
        split="train",
    )
    train_dataset = train_dataset
    train_id = train_dataset["guid"]
    pipeline = PREP_PIPELINE[data_args.prep_pipeline_name]

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    tokenizer.add_special_tokens(
        {"additional_special_tokens": list(task_infos.markers.values())}
    )
    tokenized_train_datasets = pipeline(train_dataset,
                                        tokenizer,
                                        task_infos,)

    collate_cls = COLLATOR_MAP[data_args.collator_name]
    data_collator = collate_cls(tokenizer)
    train_dataloader = torch.utils.data.DataLoader(tokenized_train_datasets,
                                    collate_fn=data_collator,
                                    batch_size=training_args.per_device_train_batch_size, shuffle=False,
                                    num_workers=4, pin_memory=True)

    checkpoint = f"./best/bert_all/fold{command_args.fold}"
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint, output_hidden_states=True)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Path Setup : analysis results will be saved in `./analysis`
    run_name = training_args.run_name
    analysis_dir = f'./analysis/{run_name}_train/fold{command_args.fold}'
    if (os.path.isdir(analysis_dir) == False):
        os.makedirs(analysis_dir)

    # Inference
    output_logit = []
    output_prob = []
    output_pred = []

    batch_size=training_args.per_device_train_batch_size
    embeddings = np.zeros((len(tokenized_train_datasets), model.config.hidden_size), dtype=np.float32)

    for i, data in enumerate(tqdm(train_dataloader)):

        with torch.no_grad():
            outputs = model(
                    input_ids=data['input_ids'].to(device),
                    attention_mask=data['attention_mask'].to(device),
                    token_type_ids=data['token_type_ids'].to(device) if 'token_type_ids' in data.keys() else None,
                    )

        hidden_state = outputs[1][-1][:, 0, :].detach().cpu().numpy()
        if len(hidden_state) == batch_size:
            embeddings[i*batch_size:(i+1)*batch_size] = hidden_state
        else:
            embeddings[i*batch_size:i*batch_size + len(hidden_state)] = hidden_state

        logits = outputs[0]
        prob = nn.functional.softmax(logits, dim=-1).detach().cpu().numpy()
        logits = logits.detach().cpu().numpy()
        result = np.argmax(logits, axis=-1)

        output_logit.append(logits)
        output_pred.append(result)
        output_prob.append(prob)

    # Save Embeddings
    np.save(os.path.join(analysis_dir, f'embeddings.npy'), embeddings)
    print('Embedding vectors saved in ' , os.path.join(analysis_dir, f'embeddings.npy'))

    # Save Confusion Matrix & Dataframe
    pred_answer = np.concatenate(output_pred).tolist()
    output_prob = np.concatenate(output_prob, axis=0).tolist()
    output = pd.DataFrame({'id':train_id, 'pred_label':pred_answer,'probs':output_prob,})
    output.to_csv(f"bert_all_fold{command_args.fold}.csv", index=False)

    # TODO After Cofusion matrix code added, clean up this code
    RELATION_CLASS = [
        'no_relation', 
        'org:top_members/employees',
        'org:members',
        'org:product',
        'per:title',
        'org:alternate_names',
        'per:employee_of',
        'org:place_of_headquarters',
        'per:product',
        'org:number_of_employees/members',
        'per:children',
        'per:place_of_residence', 
        'per:alternate_names',
        'per:other_family',
        'per:colleagues',
        'per:origin', 
        'per:siblings',
        'per:spouse',
        'org:founded',
        'org:political/religious_affiliation',
        'org:member_of',
        'per:parents',
        'org:dissolved',
        'per:schools_attended',
        'per:date_of_death', 
        'per:date_of_birth',
        'per:place_of_birth',
        'per:place_of_death',
        'org:founded_by',
        'per:religion'
    ]
    import matplotlib.pyplot as plt
    import seaborn as sns

    def get_confusion_matrix(logit_or_preds, labels, logit=False):
        preds = np.argmax(logit_or_preds, axis=1).ravel() if logit else logit_or_preds
        cm = confusion_matrix(labels, preds)
        norm_cm = cm / np.sum(cm, axis=1)[:,None]
        cm = pd.DataFrame(norm_cm, index=RELATION_CLASS, columns=RELATION_CLASS)
        fig = plt.figure(figsize=(12,9))
        sns.heatmap(cm, annot=True)
        return fig

    cm_fig = get_confusion_matrix(output['pred_label'].values, tokenized_train_datasets['label'])
    cm_fig.savefig(os.path.join(analysis_dir, f'confusion_mtx.png'), dpi=300)
    torch.save(output, os.path.join(analysis_dir, 'data_frame.pt'))
    print('Dataframe & Confusion matrix saved in ' , analysis_dir)

    print('---- Real Finish! ----')
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--fold', type=int, default=0, help='k-fold fold num: 1~5 & no k-fold: 0 (default)')
    parser.add_argument('--config', type=str, default="config/kfold.yaml", help='config file path (default: config/kfold.yaml)')
    command_args = parser.parse_args()
    print(command_args)

    main(command_args)