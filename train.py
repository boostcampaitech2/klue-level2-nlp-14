import pickle as pickle
import os
import pandas as pd
import random
import torch
import numpy as np
import argparse
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification, Trainer, TrainingArguments, RobertaConfig, RobertaTokenizer, RobertaForSequenceClassification, BertTokenizer
from transformers.configuration_utils import PretrainedConfig
from datasets import Dataset, DatasetDict, Features, Value, ClassLabel, load_dataset

from collator import DataCollator
from preprocessing import mark_entity_spans as _mark_entity_spans
from preprocessing import convert_example_to_features as _convert_example_to_features
from metrics import compute_metrics
from utils import set_seeds, softmax

from functools import partial
from typing import Tuple, List, Any, Dict

def train(args):
  # set seeds
  set_seeds(args.seed)

  # load model and tokenizer
  MODEL_NAME = args.model_name

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
  train_data = load_dataset("jinmang2/load_klue_re", script_version=args.data_ver, split='train')
  valid_data = load_dataset("jinmang2/load_klue_re", script_version=args.data_ver, split='valid')

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
  os.environ["WANDB_PROJECT"] = args.wandb_project

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
    output_dir=f'./results/{args.output_dir}',          # output directory
    save_total_limit=5,              # number of total save model.
    save_steps=500,                 # model saving step.
    num_train_epochs=args.epochs,              # total number of training epochs
    learning_rate=args.lr,               # learning_rate
    per_device_train_batch_size=args.batch_size,  # batch size per device during training
    per_device_eval_batch_size=args.valid_batch_size,   # batch size for evaluation
    warmup_ratio=args.warmup_ratio,               # number of warmup steps for learning rate scheduler
    weight_decay=args.weight_decay,               # strength of weight decay
    report_to=args.report_to,
    fp16=True,
    run_name = args.run_name,
    evaluation_strategy='steps', # evaluation strategy to adopt during training
                                # `no`: No evaluation during training.
                                # `steps`: Evaluate every `eval_steps`.
                                # `epoch`: Evaluate every end of epoch.
    eval_steps = 500,            # evaluation step.
    load_best_model_at_end = True,
    metric_for_best_model="auprc"
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
    data_collator=data_collator,
    compute_metrics=compute_metrics         # define metrics function
  )

  # train model
  trainer.train()
  model.save_pretrained(f'./best_model/{args.best_model_dir}')

  # Load Checkpoint
  config = PretrainedConfig.from_json_file(os.path.join(f'./best_model/{args.best_model_dir}', "config.json"))
  # We load the model state dict on the CPU to avoid an OOM error.
  state_dict = torch.load(os.path.join(f'./best_model/{args.best_model_dir}', "pytorch_model.bin"), map_location="cpu")
  # If the model is on the GPU, it still works!
  trainer._load_state_dict_in_model(state_dict)
  del state_dict

  # print(list(trainer.model.parameters())[-1])

  # Inference
  test_data = load_dataset("jinmang2/load_klue_re", script_version=args.data_ver, split='test')
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
  output.to_csv(f'./prediction/{args.output_name}.csv', index=False)
  #### 필수!! ##############################################
  print('---- Finish! ----')


if __name__ == '__main__':
    # args parser
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_name', type=str, default='tunib/electra-ko-small', help='model name (default: tunib/electra-ko-small)')
    parser.add_argument('--data_ver', type=str, default='v1.0.1b', help='data version (default: v1.0.1b)')
    parser.add_argument('--seed', type=int, default=42, help='random seed (default: 42)')
    parser.add_argument('--epochs', type=int, default=1, help='number of epochs to train (default: 1)')
    parser.add_argument('--batch_size', type=int, default=16, help='input batch size for training (default: 16)')
    parser.add_argument('--valid_batch_size', type=int, default=16, help='input batch size for validing (default: 16)') 
    parser.add_argument('--lr', type=float, default=5e-5, help='learning rate (default: 5e-5)')
    parser.add_argument('--warmup_ratio', type=float, default=0.2, help='warmup ratio (default: 0.2)')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='weight decay (default: 0.01)')
    parser.add_argument('--output_name', type=str, default='submission', help='submission saved name: {output_name}')

    # Wandb
    parser.add_argument('--report_to', default='wandb', help='report_to (default: wandb)')
    parser.add_argument('--wandb_project', default='klue_re_tunib_koelectra', help='wandb project name')
    parser.add_argument('--run_name', default='baseline', help='run_name')

    # Container environment
    parser.add_argument('--output_dir', type=str, default=os.environ.get('MODEL_DIR', 'result1'), help='checkpoint models saved at (default: f"./results/{best_model_dir}")')
    parser.add_argument('--best_model_dir', type=str, default=os.environ.get('BEST_MODEL_DIR', 'model1'), help='best model saved at (default: f"./best_model/{best_model_dir}")')

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
