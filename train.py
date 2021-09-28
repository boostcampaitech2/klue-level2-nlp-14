import pickle as pickle
import os
import pandas as pd
from functools import partial
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import sklearn
import numpy as np
import random
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification, Trainer, TrainingArguments, RobertaConfig, RobertaTokenizer, RobertaForSequenceClassification, BertTokenizer
from datasets import load_dataset
import argparse
import wandb
from load_klue_re import KlueRE
from preprocessing import mark_entity_spans as _mark_entity_spans
from preprocessing import convert_example_to_features as _convert_example_to_features
from metrics import make_compute_metrics
from collator import DataCollator
from utils import softmax


def seed_everything(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def klue_re_micro_f1(preds, labels):
    """KLUE-RE micro f1 (except no_relation)"""
    label_list = ['no_relation', 'org:top_members/employees', 'org:members',
       'org:product', 'per:title', 'org:alternate_names',
       'per:employee_of', 'org:place_of_headquarters', 'per:product',
       'org:number_of_employees/members', 'per:children',
       'per:place_of_residence', 'per:alternate_names',
       'per:other_family', 'per:colleagues', 'per:origin', 'per:siblings',
       'per:spouse', 'org:founded', 'org:political/religious_affiliation',
       'org:member_of', 'per:parents', 'org:dissolved',
       'per:schools_attended', 'per:date_of_death', 'per:date_of_birth',
       'per:place_of_birth', 'per:place_of_death', 'org:founded_by',
       'per:religion']
    no_relation_label_idx = label_list.index("no_relation")
    label_indices = list(range(len(label_list)))
    label_indices.remove(no_relation_label_idx)
    return sklearn.metrics.f1_score(labels, preds, average="micro", labels=label_indices) * 100.0


def klue_re_auprc(probs, labels):
    """KLUE-RE AUPRC (with no_relation)"""
    labels = np.eye(30)[labels]

    score = np.zeros((30,))
    for c in range(30):
        targets_c = labels.take([c], axis=1).ravel()
        preds_c = probs.take([c], axis=1).ravel()
        precision, recall, _ = sklearn.metrics.precision_recall_curve(targets_c, preds_c)
        score[c] = sklearn.metrics.auc(recall, precision)
    return np.average(score) * 100.0


def compute_metrics(pred):
    """ validation을 위한 metrics function """
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    probs = pred.predictions

    # calculate accuracy using sklearn's function
    f1 = klue_re_micro_f1(preds, labels)
    auprc = klue_re_auprc(probs, labels)
    acc = accuracy_score(labels, preds) # 리더보드 평가에는 포함되지 않습니다.

    return {
        'micro f1 score': f1,
        'auprc' : auprc,
        'accuracy': acc,
    }


def train(args):
    seed_everything(args.seed)

    MODEL_NAME = "klue/bert-base"
    #MODEL_NAME = "klue/roberta-large"
    NUM_LABELS = KlueRE.BUILDER_CONFIGS[0].features["label"].num_classes

    # load dataset
    dataset = load_dataset("jinmang2/load_klue_re", script_version="v1.0.1b")
    train_data = dataset['train']
    valid_data = dataset['valid']

    # load tokenizer
    markers = dict(
        subject_start_marker="<subj>",
        subject_end_marker="</subj>",
        object_start_marker="<obj>",
        object_end_marker="</obj>",
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
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

    # tokenizing dataset
    train_examples = train_data.map(mark_entity_spans)
    tokenized_train_datasets = train_examples.map(convert_example_to_features)
    valid_examples = valid_data.map(mark_entity_spans)
    tokenized_valid_datasets = valid_examples.map(convert_example_to_features)

    # setting model hyperparameter
    relation_class = KlueRE.BUILDER_CONFIGS[0].features["label"].names
    id2label = {idx: label for idx, label in enumerate(relation_class)}
    label2id = {label: idx for idx, label in enumerate(relation_class)}
    model_config = AutoConfig.from_pretrained(
        pretrained_model_name_or_path=MODEL_NAME,
        num_labels=NUM_LABELS,
        cache_dir="cache",
        id2label=id2label,
        label2id=label2id,
    )

    # load model
    model = AutoModelForSequenceClassification.from_pretrained(
        pretrained_model_name_or_path=MODEL_NAME,
        config=model_config,
        cache_dir="cache",
    )
    print(model.config)

    if model.config.vocab_size < len(tokenizer):
        print("resize...")
        model.resize_token_embeddings(len(tokenizer))
    
    # Load metrics and collator
    #compute_metrics = make_compute_metrics(relation_class)
    data_collator = DataCollator(tokenizer)

    # setting device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)
    model.to(device)
  
    # Build huggingface Trainer
    training_args = TrainingArguments(
        output_dir=f'./results/{args.run_name}',          # output directory
        save_total_limit=5,              # number of total save model.
        save_steps=500,                 # model saving step.
        num_train_epochs=args.epochs,              # total number of training epochs
        learning_rate=args.lr,               # learning_rate
        per_device_train_batch_size=args.batch_size,  # batch size per device during training
        per_device_eval_batch_size=args.valid_batch_size,   # batch size for evaluation
        warmup_steps=500,                # number of warmup steps for learning rate scheduler
        weight_decay=args.weight_decay,               # strength of weight decay
        logging_dir='./logs',            # directory for storing logs
        logging_steps=100,              # log saving step.
        evaluation_strategy='steps', # evaluation strategy to adopt during training
                                    # `no`: No evaluation during training.
                                    # `steps`: Evaluate every `eval_steps`.
                                    # `epoch`: Evaluate every end of epoch.
        eval_steps = 500,            # evaluation step.
        load_best_model_at_end = True,
        report_to=args.report_to,
        run_name=args.run_name
    )

    # remove unused feature names
    features_name = list(tokenized_train_datasets.features.keys())
    features_name.pop(features_name.index("input_ids"))
    features_name.pop(features_name.index("label"))
    tokenized_train_datasets = tokenized_train_datasets.remove_columns(features_name)
    tokenized_valid_datasets = tokenized_valid_datasets.remove_columns(features_name)
    
    #optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    #scheduler = CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-6)

    trainer = Trainer(
        model=model,                         # the instantiated 🤗 Transformers model to be trained
        args=training_args,                  # training arguments, defined above
        train_dataset=tokenized_train_datasets,         # training dataset
        eval_dataset=tokenized_valid_datasets,             # evaluation dataset
        compute_metrics=compute_metrics,         # define metrics function
        data_collator=data_collator,
        #optimizers=(optimizer, scheduler)
    )

    # train model
    trainer.train()
    trainer.model.save_pretrained(f'./results/{args.run_name}/best_model')

    # Inference
    test_data = dataset['test']
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
    output.to_csv(f'./prediction/{args.run_name}.csv', index=False)
    #### 필수!! ##############################################
    print('---- Finish! ----')



if __name__ == '__main__':
    # args parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42, help='random seed (default: 42)')
    parser.add_argument('--epochs', type=int, default=5, help='number of epochs to train (default: 5)')
    parser.add_argument('--batch_size', type=int, default=32, help='input batch size for training (default: 16)')
    parser.add_argument('--valid_batch_size', type=int, default=32, help='input batch size for validing (default: 16)') 
    parser.add_argument('--val_ratio', type=float, default=0.2, help='ratio for validaton (default: 0.2)')
    parser.add_argument('--lr', type=float, default=4e-5, help='learning rate (default: 5e-5)')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='weight decay (default: 0.01)')
    parser.add_argument('--report_to', default='wandb', help='report_to (default: wandb)')
    parser.add_argument('--wandb_project', default='klue_re_bert_hrlee', help='wandb project name')
    parser.add_argument('--run_name', default='baseline', help='run_name')
    args = parser.parse_args()
    print(args)

    # wandb setting
    os.environ["WANDB_PROJECT"] = args.wandb_project
    call_wandb = True
    try:
        os.environ["WANDB_PROJECT"]
    
    except KeyError:
        call_wandb = False
    if call_wandb:
        import wandb
        wandb.login()

    # training
    train(args)
