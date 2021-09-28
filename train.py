import pickle as pickle
import os
import pandas as pd
import random
import torch
import sklearn
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification, Trainer, TrainingArguments, RobertaConfig, RobertaTokenizer, RobertaForSequenceClassification, BertTokenizer
from transformers.configuration_utils import PretrainedConfig
from load_data import *
from datasets import Dataset, DatasetDict, Features, Value, ClassLabel, load_dataset

from collator import DataCollator
from preprocessing import mark_entity_spans as _mark_entity_spans
from preprocessing import convert_example_to_features as _convert_example_to_features

from functools import partial
from typing import Tuple, List, Any, Dict

# set fixed random seed
def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

def softmax(arr: np.ndarray, axis: int = -1):
    c = arr.max(axis=axis, keepdims=True)
    s = arr - c
    nominator = np.exp(s)
    denominator = nominator.sum(axis=axis, keepdims=True)
    probs = nominator / denominator
    return probs

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

def label_to_num(label):
  num_label = []
  with open('dict_label_to_num.pkl', 'rb') as f:
    dict_label_to_num = pickle.load(f)
  for v in label:
    num_label.append(dict_label_to_num[v])
  
  return num_label

def train():

  wandb_project = "klue_re_tunib_koelectra_2"
  report_to = "wandb"

  # load model and tokenizer
  # MODEL_NAME = "bert-base-uncased"
  # MODEL_NAME = "klue/bert-base"
  seed_everything(42)
  MODEL_NAME = "tunib/electra-ko-small"

    # Settings for Relation Extraction Baseline
    # <subj>entity</subj> ~~ <obj>entity</obj> ~~
  markers = dict(
      subject_start_marker="<subj>",
      subject_end_marker="</subj>",
      object_start_marker="<obj>",
      object_end_marker="</obj>",
  )

  relation_class = ['no_relation', 'org:top_members/employees', 'org:members',
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

  num_labels = len(relation_class)

  id2label = {idx: label for idx, label in enumerate(relation_class)}
  label2id = {label: idx for idx, label in enumerate(relation_class)}
  
  # Load Tokenizer
  tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
  tokenizer.add_special_tokens(
        {"additional_special_tokens": list(markers.values())}
  )

  # load dataset
  train_data = load_dataset("jinmang2/load_klue_re", script_version="v1.0.1b", split='train')
  valid_data = load_dataset("jinmang2/load_klue_re", script_version="v1.0.1b", split='valid')

  # Preprocess and tokenizing
  mark_entity_spans = partial(_mark_entity_spans, **markers)
  convert_example_to_features = partial(
      _convert_example_to_features,
      tokenizer=tokenizer,
      **markers,
  )

  train_examples = train_data.map(mark_entity_spans)
  tokenized_train_datasets = train_examples.map(convert_example_to_features)
  valid_examples = valid_data.map(mark_entity_spans)
  tokenized_valid_datasets = valid_examples.map(convert_example_to_features)

  # make dataset for pytorch.
  # RE_train_dataset = RE_Dataset(tokenized_train, train_label)
  # RE_dev_dataset = RE_Dataset(tokenized_dev, dev_label)
  
  device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

  print(device)
  # setting model hyperparameter
  model_config = AutoConfig.from_pretrained(
        pretrained_model_name_or_path=MODEL_NAME,
        num_labels=num_labels,
        cache_dir="cache",
        id2label=id2label,
        label2id=label2id,
    )
  # model_config =  AutoConfig.from_pretrained(MODEL_NAME)
  # model_config.num_labels = 30

  model =  AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, config=model_config, cache_dir="cache")
  print(model.config)
  model.parameters
  model.to(device)
  
  if model.config.vocab_size < len(tokenizer):
    print("resize...")
    model.resize_token_embeddings(len(tokenizer))

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

  # 사용한 option 외에도 다양한 option들이 있습니다.
  # https://huggingface.co/transformers/main_classes/trainer.html#trainingarguments 참고해주세요.
  training_args = TrainingArguments(
    output_dir='./results/0928_tunib_electra',          # output directory
    save_total_limit=5,              # number of total save model.
    save_steps=500,                 # model saving step.
    num_train_epochs=10,              # total number of training epochs
    learning_rate=5e-5,               # learning_rate
    per_device_train_batch_size=32,  # batch size per device during training
    per_device_eval_batch_size=32,   # batch size for evaluation
    warmup_steps = 500,               # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_dir='./0928_logs',            # directory for storing logs
    logging_steps=100,              # log saving step.
    report_to=report_to,
    run_name = "baseline-0928-taeukkkim",
    evaluation_strategy='steps', # evaluation strategy to adopt during training
                                # `no`: No evaluation during training.
                                # `steps`: Evaluate every `eval_steps`.
                                # `epoch`: Evaluate every end of epoch.
    eval_steps = 500,            # evaluation step.
    load_best_model_at_end = True 
  )

  # remove unused feature names
  features_name = list(tokenized_train_datasets.features.keys())
  features_name.pop(features_name.index("input_ids"))
  features_name.pop(features_name.index("label"))
  tokenized_train_datasets = tokenized_train_datasets.remove_columns(features_name)
  tokenized_valid_datasets = tokenized_valid_datasets.remove_columns(features_name)

  trainer = Trainer(
    model=model,                         # the instantiated 🤗 Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=tokenized_train_datasets,         # training dataset
    eval_dataset=tokenized_valid_datasets,             # evaluation dataset
    data_collator = data_collator,
    compute_metrics=compute_metrics         # define metrics function
  )

  # train model
  trainer.train()
  model.save_pretrained('./0928_best_model')

  # Load Checkpoint
  config = PretrainedConfig.from_json_file(os.path.join('./0928_best_model', "config.json"))
  # We load the model state dict on the CPU to avoid an OOM error.
  state_dict = torch.load(os.path.join('./0928_best_model', "pytorch_model.bin"), map_location="cpu")
  # If the model is on the GPU, it still works!
  trainer._load_state_dict_in_model(state_dict)
  del state_dict

  # print(list(trainer.model.parameters())[-1])

  # Inference
  test_data = load_dataset("jinmang2/load_klue_re", script_version="v1.0.1b", split='test')
  test_id = test_data["guid"]
  examples = test_data.map(mark_entity_spans)
  tokenized_test_datasets = examples.map(convert_example_to_features)

  features_name = list(tokenized_test_datasets.features.keys())
  features_name.pop(features_name.index("input_ids"))

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
  output.to_csv(f'./prediction/submission_0928_tunib_electra_small.csv', index=False)
  #### 필수!! ##############################################
  print('---- Finish! ----')


def main():
  train()

if __name__ == '__main__':
  main()
