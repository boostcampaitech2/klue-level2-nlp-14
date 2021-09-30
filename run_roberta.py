import os

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

from load_klue_re import KlueRE
from preprocessing import mark_entity_spans as _mark_entity_spans
from preprocessing import convert_example_to_features as _convert_example_to_features
from metrics import make_compute_metrics
from collator import DataCollator
from utils import softmax


def main():

    # Settings
    # @TODO: argparse로 만들기
    learning_rate = 3e-05
    num_train_epochs = 4
    train_batch_size = 32
    eval_batch_size = 32
    warmup_ratio = 0.2
    # patience = 10000
    output_dir = "klue_dir"
    wandb_project = "klue_re"
    run_name = "baseline-0927-jinmang2-2"
    report_to = "wandb"
    submit = f"submission-{run_name}" + "test"
    best_model_path = "0927_best"

    model_name_or_path = "klue/roberta-large"
    
    # Settings for Relation Extraction Baseline
    # <subj>entity</subj> ~~ <obj>entity</obj> ~~
    markers = dict(
        subject_start_marker="<subj>",
        subject_end_marker="</subj>",
        object_start_marker="<obj>",
        object_end_marker="</obj>",
    )

    relation_class = KlueRE.BUILDER_CONFIGS[0].features["label"].names
    num_labels = KlueRE.BUILDER_CONFIGS[0].features["label"].num_classes

    id2label = {idx: label for idx, label in enumerate(relation_class)}
    label2id = {label: idx for idx, label in enumerate(relation_class)}
    
    # Get training data
    train_data = load_dataset("load_klue_re.py", split="train")
    # valid_data = load_dataset("klue", "re", split="validation")
    # from o2n import orig2new
    # valid_data = valid_data.map(orig2new)
    
    # Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    tokenizer.add_special_tokens(
        {"additional_special_tokens": list(markers.values())}
    )
    
    # Preprocess and tokenizing
    mark_entity_spans = partial(_mark_entity_spans, **markers)
    convert_example_to_features = partial(
        _convert_example_to_features,
        tokenizer=tokenizer,
        **markers,
    )

    train_examples = train_data.map(mark_entity_spans)
    tokenized_train_datasets = train_examples.map(convert_example_to_features)
    # valid_examples = valid_data.map(mark_entity_spans)
    # tokenized_valid_datasets = valid_examples.map(convert_example_to_features)
    
    # Get model (RoBERTa-Large)
    config = AutoConfig.from_pretrained(
        pretrained_model_name_or_path=model_name_or_path,
        num_labels=num_labels,
        cache_dir="cache",
        id2label=id2label,
        label2id=label2id,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        pretrained_model_name_or_path=model_name_or_path,
        config=config,
        cache_dir="cache",
    )
    
    if model.config.vocab_size < len(tokenizer):
        print("resize...")
        model.resize_token_embeddings(len(tokenizer))
    
    # Load metrics and collator
    compute_metrics = make_compute_metrics(relation_class)
    data_collator = DataCollator(tokenizer)
    
    # Set-up WANDB
    os.environ["WANDB_PROJECT"] = wandb_project

    call_wandb = True
    try:
        os.environ["WANDB_PROJECT"]

    except KeyError:
        call_wandb = False

    if call_wandb:
        import wandb
        wandb.login()
    
    # Build huggingface Trainer
    args = TrainingArguments(
        output_dir=output_dir,
        # evaluation_strategy="epoch",
        # save_strategy="epoch",
        learning_rate=learning_rate,
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=eval_batch_size,
        weight_decay=0.01,
        warmup_ratio=warmup_ratio,
    #     save_total_limit=5,
        num_train_epochs=num_train_epochs,
        fp16=True,
        report_to=report_to,
        run_name=run_name,
        load_best_model_at_end=True,
        metric_for_best_model="auprc",
    )
    
    # remove unused feature names
    features_name = list(tokenized_train_datasets.features.keys())
    features_name.pop(features_name.index("input_ids"))
    features_name.pop(features_name.index("label"))
    tokenized_train_datasets = tokenized_train_datasets.remove_columns(features_name)
    # tokenized_valid_datasets = tokenized_valid_datasets.remove_columns(features_name)

    # ====================================
    # select few samples
    # from datasets import DatasetDict

    # tokenized_datasets = DatasetDict(
    #     {
    #         "train": tokenized_train_datasets.select(range(400)),
    #         # "validation": tokenized_datasets["validation"].select(range(1000)),
    #     }
    # )
    # tokenized_train_datasets = tokenized_datasets["train"]
    # ====================================

    trainer = Trainer(
        model,
        args,
        train_dataset=tokenized_train_datasets,
        # eval_dataset=tokenized_valid_datasets,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    
    # Training
    trainer.train()
    trainer.model.save_pretrained(best_model_path)
    
    # Load Checkpoint
    config = PretrainedConfig.from_json_file(os.path.join(best_model_path, "config.json"))
    # We load the model state dict on the CPU to avoid an OOM error.
    state_dict = torch.load(os.path.join(best_model_path, "pytorch_model.bin"), map_location="cpu")
    # If the model is on the GPU, it still works!
    trainer._load_state_dict_in_model(state_dict)
    del state_dict

    # print(list(trainer.model.parameters())[-1])

    # Inference
    test_data = load_dataset("load_klue_re.py", split="test")
    test_id = test_data["guid"]
    examples = test_data.map(mark_entity_spans)
    tokenized_test_datasets = examples.map(convert_example_to_features)

    features_name = list(tokenized_test_datasets.features.keys())
    features_name.pop(features_name.index("input_ids"))
    # features_name.pop(features_name.index("label"))
    tokenized_test_datasets = tokenized_test_datasets.remove_columns(features_name)

    logits = trainer.predict(tokenized_test_datasets)[0]
    probs = softmax(logits).tolist()
    result = np.argmax(logits, axis=-1).tolist()
    pred_answer = [id2label[v] for v in result]

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
    output.to_csv(f'./prediction/{submit}.csv', index=False)
    #### 필수!! ##############################################
    print('---- Finish! ----')

if __name__ == "__main__":
    main()