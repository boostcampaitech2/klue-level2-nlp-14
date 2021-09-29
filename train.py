import pickle as pickle
import os
import pandas as pd
import torch
import sklearn
import wandb
import argparse
import numpy as np
from utils import softmax, seed_everything
from metrics import compute_metrics
from config import label_list, device, markers
from functools import partial
from preprocessing import mark_entity_spans as _mark_entity_spans
from preprocessing import convert_example_to_features as _convert_example_to_features
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification, Trainer, TrainingArguments, RobertaConfig, RobertaTokenizer, RobertaForSequenceClassification, BertTokenizer
from load_data import *
from imbalanced_sampler_trainer import ImbalancedSamplerTrainer
from datasets import load_dataset
from collator import DataCollator

# def label_to_num(label):
#     num_label = []
#     with open('dict_label_to_num.pkl', 'rb') as f:
#         dict_label_to_num = pickle.load(f)
#     for v in label:
#         num_label.append(dict_label_to_num[v])
  
#     return num_label

def train(args):
    # load model and tokenizer
    # MODEL_NAME = "bert-base-uncased"
    seed_everything()

    MODEL_NAME = args.model_name
    wandb_project = args.wandb_project
    report_to = "wandb"

    relation_class = label_list
    num_labels = 30

    id2label = {idx: label for idx, label in enumerate(relation_class)}
    label2id = {label: idx for idx, label in enumerate(relation_class)}

    train_data = load_dataset("jinmang2/load_klue_re", script_version="v1.0.1b")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.add_special_tokens(
        {"additional_special_tokens" : list(markers.values())}
    )

    mark_entity_spans = partial(_mark_entity_spans, **markers)
    convert_example_to_features = partial(
        _convert_example_to_features,
        tokenizer=tokenizer,
        **markers,
    )

    train_examples = train_data['train'].map(mark_entity_spans)
    eval_examples = train_data['valid'].map(mark_entity_spans)

    tokenized_train_datasets = train_examples.map(convert_example_to_features)
    tokenized_eval_datasets = eval_examples.map(convert_example_to_features)
    # load dataset
    #train_dataset, eval_dataset = load_data("../dataset/train/train.csv")
    

    #train_label = label_to_num(train_dataset['label'].values)
    #eval_label = label_to_num(eval_dataset['label'].values)

    # tokenizing dataset
    #tokenized_train = tokenized_dataset(train_dataset, tokenizer)
    #tokenized_eval = tokenized_dataset(eval_dataset, tokenizer)

    # make dataset for pytorch.
    #RE_train_dataset = RE_Dataset(tokenized_train, train_dataset['label'])
    #RE_eval_dataset = RE_Dataset(tokenized_eval, eval_dataset['label'])

    
    print(device)
    # setting model hyperparameter
    model_config =  AutoConfig.from_pretrained(
        pretrained_model_name_or_path=MODEL_NAME, 
        cache_dir="cache",
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id
    )

    model =  AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, config=model_config, cache_dir="cache").to(device)
 
    if model.config.vocab_size < len(tokenizer):
        print("resize...")
        model.resize_token_embeddings(len(tokenizer))
    
    # collator
    data_collator = DataCollator(tokenizer)

    # Set-up WANDB
    os.environ["WANDB_PROJECT"] = wandb_project

    # 사용한 option 외에도 다양한 option들이 있습니다.
    # https://huggingface.co/transformers/main_classes/trainer.html#trainingarguments 참고해주세요.
    training_args = TrainingArguments(
        output_dir=args.output_dir,          # output directory
        save_total_limit=5,              # number of total save model.
        save_steps=args.save_steps,                 # model saving step.
        num_train_epochs=args.epochs,              # total number of training epochs
        learning_rate=args.lr,               # learning_rate
        per_device_train_batch_size=args.train_batch,  # batch size per device during training
        per_device_eval_batch_size=args.valid_batch,   # batch size for evaluation
        warmup_steps=args.warmup_steps,                # number of warmup steps for learning rate scheduler
        weight_decay=args.weight_decay,               # strength of weight decay
        logging_dir='./logs',            # directory for storing logs
        logging_steps=100,              # log saving step.
        evaluation_strategy='steps', # evaluation strategy to adopt during training
                                    # `no`: No evaluation during training.
                                    # `steps`: Evaluate every `eval_steps`.
                                    # `epoch`: Evaluate every end of epoch.
        eval_steps = 500,            # evaluation step.
        report_to = report_to,
        load_best_model_at_end = True 
      )

    # remove unused feature names
    features_name = list(tokenized_train_datasets.features.keys())
    features_name.pop(features_name.index("input_ids"))
    features_name.pop(features_name.index("label"))
    tokenized_train_datasets = tokenized_train_datasets.remove_columns(features_name)

    trainer = ImbalancedSamplerTrainer(
        model=model,                         # the instantiated 🤗 Transformers model to be trained
        args=training_args,                  # training arguments, defined above
        train_dataset=tokenized_train_datasets,         # training dataset
        eval_dataset=tokenized_eval_datasets,             # evaluation dataset
        compute_metrics=compute_metrics,         # define metrics function
        data_collator=data_collator
      )
    
      # train model
    trainer.train()
    model.save_pretrained('./best_model')

    test_data = load_dataset("jinmang2/load_klue_re", script_version="v1.0.1b", split="test")
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
    output.to_csv(f'./prediction/{args.submit}.csv', index=False)
    #### 필수!! ##############################################
    print('---- Finish! ----')

def main(args):
    train(args)
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--output_dir', type=str, default='baseline_0928_jeangyu',help='output directory')
    parser.add_argument('--save_steps', type=int, default=500,help='number of steps to save (default: 1)')
    parser.add_argument('--epochs', type=int, default=5,help='number of epochs to train (default: 1)')
    parser.add_argument('--lr', type=float, default=3e-5,help='learning rate (default: 3e-5)')
    parser.add_argument('--train_batch', type=int, default=32,help='batch size per device during training')
    parser.add_argument('--valid_batch', type=int, default=32,help='batch size for evaluation')
    parser.add_argument('--warmup_steps', type=int, default=500,help='number of warmup steps for learning rate scheduler')
    parser.add_argument('--weight_decay', type=float, default=0.01,help='strength of weight decay')
    parser.add_argument('--model_name', type=str, default='monologg/koelectra-base-v3-discriminator',help='model name')
    parser.add_argument('--wandb_project', default='klue_re_KoElectra_base_jgyu', help='wandb project name')
    parser.add_argument('--submit', default='submission_0', help='submission file name')

    args = parser.parse_args()

    main(args)    