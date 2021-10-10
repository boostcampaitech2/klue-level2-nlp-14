# run_xlm.py
"""
Execution file for training for xlm model. (The baseline provided in the competition.)
"""

import os
import torch
import argparse
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification, Trainer, TrainingArguments, RobertaConfig, RobertaTokenizer, RobertaForSequenceClassification, BertTokenizer
from solution.data.load_data import *
import wandb
from solution.utils import compute_metrics, set_seeds
from sklearn.model_selection import StratifiedKFold
from solution.trainers import XLMTrainer

from datasets import load_dataset, Dataset, DatasetDict, Features, Value, ClassLabel, concatenate_datasets

def train():
    MODEL_NAME = "xlm-roberta-large"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # load dataset    
    raw_dataset = load_dataset("jinmang2/load_klue_re", script_version="v3.0.1")
    concat = concatenate_datasets([raw_dataset['train'], raw_dataset['aug1']])
    concat_df = Dataset.to_pandas(concat)
    
    train_df = Dataset.to_pandas(raw_dataset['train'])
    
    concat_df['subject_entity'] = concat_df['subject_entity'].map(str)
    concat_df['object_entity'] = concat_df['object_entity'].map(str)
    concat_df['guid'] = range(len(concat_df))
    
    train_df['subject_entity'] = train_df['subject_entity'].map(str)
    train_df['object_entity'] = train_df['object_entity'].map(str)
    train_df['guid'] = range(len(train_df))
    
    train_dataset = concat_df
    dev_dataset = train_df

    train_label = concat_df['label'].values
    dev_label = train_df['label'].values

    # tokenizing dataset
    tokenized_train = tokenized_dataset(train_dataset, tokenizer)
    tokenized_dev = tokenized_dataset(dev_dataset, tokenizer)

    # make dataset for pytorch.
    RE_train_dataset = RE_Dataset(tokenized_train, train_label)
    RE_dev_dataset = RE_Dataset(tokenized_dev, dev_label)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    print(device)
    # setting model hyperparameter
    model_config =  AutoConfig.from_pretrained(MODEL_NAME)
    model_config.num_labels = 30

    model =  AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, config=model_config)
    print(model.config)
    model.parameters
    model.to(device)

    # 사용한 option 외에도 다양한 option들이 있습니다.
    # https://huggingface.co/transformers/main_classes/trainer.html#trainingarguments 참고해주세요.
    training_args = TrainingArguments(
        output_dir='./results',          # output directory
        save_total_limit=5,              # number of total save model.
        save_steps=500,                  # model saving step.
        num_train_epochs=3,              # total number of training epochs
        learning_rate=3e-5,              # learning_rate
        per_device_train_batch_size=64,  # batch size per device during training
        per_device_eval_batch_size=64,   # batch size for evaluation
        warmup_ratio=0.1,
        warmup_steps=500,                # number of warmup steps for learning rate scheduler
        weight_decay=0.01,               # strength of weight decay
        logging_dir='./logs',            # directory for storing logs
        logging_steps=100,               # log saving step.
        evaluation_strategy='steps', # evaluation strategy to adopt during training
                                    # `no`: No evaluation during training.
                                    # `steps`: Evaluate every `eval_steps`.
                                    # `epoch`: Evaluate every end of epoch.
        eval_steps = 500,            # evaluation step.
        load_best_model_at_end = True,
        seed = 42,
        
        report_to = "wandb",
        run_name = f"1007-{MODEL_NAME}-final",
        metric_for_best_model = "loss",
    )
    trainer = XLMTrainer(
        model=model,                         # the instantiated 🤗 Transformers model to be trained
        args=training_args,                  # training arguments, defined above
        train_dataset=RE_train_dataset,         # training dataset
        eval_dataset=RE_dev_dataset,             # evaluation dataset
        compute_metrics=compute_metrics         # define metrics function
    )

    # train model
    trainer.train()
    model.save_pretrained(f'./best_model/{MODEL_NAME}_1007_final')


def train_kfold():
    # load model and tokenizer
    # MODEL_NAME = "bert-base-uncased"
    MODEL_NAME = "xlm-roberta-large"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # load dataset    
    raw_dataset_2 = load_dataset("jinmang2/load_klue_re", script_version="v3.0.1")
    concat = concatenate_datasets([raw_dataset_2['train'], raw_dataset_2['aug1']])
    raw_df = Dataset.to_pandas(concat)
    raw_df['subject_entity'] = raw_df['subject_entity'].map(str)
    raw_df['object_entity'] = raw_df['object_entity'].map(str)
    raw_df['guid'] = range(len(raw_df))
    
    skf = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)
    fold_data = skf.split(raw_df, raw_df['label'].values)
    
    for fold_i, (trn_idx, dev_idx) in enumerate(fold_data):
        if fold_i == iter-1:
            break
        
    train_dataset = raw_df.iloc[trn_idx]
    dev_dataset = raw_df.iloc[dev_idx]

    train_label = train_dataset['label'].values
    dev_label = dev_dataset['label'].values

    # tokenizing dataset
    tokenized_train = tokenized_dataset(train_dataset, tokenizer)
    tokenized_dev = tokenized_dataset(dev_dataset, tokenizer)

    # make dataset for pytorch.
    RE_train_dataset = RE_Dataset(tokenized_train, train_label)
    RE_dev_dataset = RE_Dataset(tokenized_dev, dev_label)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    print(device)
    # setting model hyperparameter
    model_config =  AutoConfig.from_pretrained(MODEL_NAME)
    model_config.num_labels = 30

    model =  AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, config=model_config)
    print(model.config)
    model.parameters
    model.to(device)

    # 사용한 option 외에도 다양한 option들이 있습니다.
    # https://huggingface.co/transformers/main_classes/trainer.html#trainingarguments 참고해주세요.
    training_args = TrainingArguments(
        output_dir='./results',          # output directory
        save_total_limit=5,              # number of total save model.
        save_steps=500,                  # model saving step.
        num_train_epochs=5,              # total number of training epochs
        learning_rate=3e-5,              # learning_rate
        per_device_train_batch_size=64,  # batch size per device during training
        per_device_eval_batch_size=64,   # batch size for evaluation
        warmup_ratio=0.1,
        warmup_steps=500,                # number of warmup steps for learning rate scheduler
        weight_decay=0.01,               # strength of weight decay
        logging_dir='./logs',            # directory for storing logs
        logging_steps=100,               # log saving step.
        evaluation_strategy='steps', # evaluation strategy to adopt during training
                                    # `no`: No evaluation during training.
                                    # `steps`: Evaluate every `eval_steps`.
                                    # `epoch`: Evaluate every end of epoch.
        eval_steps = 500,            # evaluation step.
        load_best_model_at_end = True,
        seed = 42,
        
        report_to = "wandb",
        run_name = f"1006-{MODEL_NAME}-{iter}/5_fold",
        metric_for_best_model = "loss",
    )
    trainer = XLMTrainer(
    # trainer = Trainer(
        model=model,                         # the instantiated 🤗 Transformers model to be trained
        args=training_args,                  # training arguments, defined above
        train_dataset=RE_train_dataset,         # training dataset
        eval_dataset=RE_dev_dataset,             # evaluation dataset
        compute_metrics=compute_metrics         # define metrics function
    )

    # train model
    trainer.train()
    model.save_pretrained(f'./best_model/{MODEL_NAME}_fold_{iter}/5')
        

if __name__ == '__main__':
    set_seeds(42)
    for iter in range(5):
        iter += 1
        print(f'full data train 1006_{iter}')
        os.environ["WANDB_PROJECT"] = "klue_re_xlm-roberta-large"
        call_wandb = True
        try:
            os.environ["WANDB_PROJECT"]
        except KeyError:
            call_wandb = False
        if call_wandb:
            import wandb
            wandb.login()
            
        # train()
        train_kfold(iter)
        

if __name__ == '__main__':
    set_seeds(42)
    print(f'full data train 1007')

    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', default='normal', type=str, help='choose mode (normal, fold)')

    args=parser.parse_args()

    os.environ["WANDB_PROJECT"] = "klue_re_xlm-roberta-large"
    call_wandb = True
    try:
        os.environ["WANDB_PROJECT"]
    except KeyError:
        call_wandb = False
    if call_wandb:
        import wandb
        wandb.login()

    if args.mode=='normal':    
        train()
    elif args.mode=='fold':
        train_kfold()
    else:
        print('choose correct model !')