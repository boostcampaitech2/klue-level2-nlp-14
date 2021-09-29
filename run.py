import os
import pickle
import argparse
import logging
import sys
import random

import numpy as np
import pandas as pd
import torch

from transformers import AutoConfig, \
    AutoModelForSequenceClassification, AutoTokenizer, Trainer, \
    TrainingArguments
import datasets

from constant import *
import preps.entity_tagging as prep
from preps.entity_tagging import *
from utils import add_general_args, inference, num_to_label, compute_metrics

# TODO Add logger
def main(args):

    # Constants in constant.py
    # CLASS_NAMES
    # N_CLASS
    # IDX2LABEL
    # LABEL2IDX

    # Basic Setup 
    MODEL_NAME = args.model_name_or_path
    CACHE_DIR = os.path.abspath(os.path.join(os.getcwd(), args.cache_dir))
    DATASET_VER = args.dataset_ver
    MAX_LEN = args.max_seq_length
    SEED = args.seed

    OUTPUT_DIR = os.path.abspath(os.path.join(os.getcwd(),args.output_dir))
    WANDB_PROJECT = args.wandb_project
    WANDB_RUN_NAME = args.wandb_run_name

    MODEL_DIR = os.path.abspath(os.path.join(os.getcwd(), 'models'))
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print("device : ", device)

    # Fix random seed
    def set_seeds(seed=42):
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False # for faster training, but not deterministic

    set_seeds(SEED)
    
    # Setting tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR)
    tokenizer.add_special_tokens(
    {"additional_special_tokens": list(MARKERS.values())}
    )
    # Load Dataset
    raw_dataset = datasets.load_dataset("jinmang2/load_klue_re", script_version=DATASET_VER, cache_dir=CACHE_DIR)
    
    # TODO Add feature to select method
    # Preprocessing
    convert_example_to_features = partial(prep.convert_example_to_features,
                                        tokenizer=tokenizer,
                                        )
    
    examples = raw_dataset.map(prep.mark_entity_spans)
    tokenized_datasets = examples.map(convert_example_to_features)
    del examples

    data_collator = DataCollator(tokenizer, max_length=MAX_LEN)

    # Setting model hyperparameter
    model_config = AutoConfig.from_pretrained(MODEL_NAME)
    model_config.num_labels = N_CLASS
    model_config.id2label = IDX2LABEL
    model_config.label2id = LABEL2IDX
    
    # Loading the pre-trained model
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        config=model_config,
        cache_dir=CACHE_DIR
    )
    
    # Checking the vocab size
    if model.config.vocab_size < len(tokenizer):
        print("Resize Token Embeddings")
        model.resize_token_embeddings(len(tokenizer))
        # model.save_pretrained(MODEL_DIR)
    
    model_config.vocab_size = len(tokenizer)
    print(model.config)
    model.to(device)

    # Wandb setup
    call_wandb = False 
    if args.report_to == 'wandb':
        os.environ["WANDB_PROJECT"] = WANDB_PROJECT
        call_wandb = True
        try:
            os.environ["WANDB_PROJECT"]

        except KeyError:
            call_wandb = False
        
        if call_wandb:
            import wandb
            wandb.login()
            wandb.init(project=WANDB_PROJECT,
            entity='kiyoung2',
            name=WANDB_RUN_NAME,
            )

    # TODO: Add arguments
    training_args = TrainingArguments(
                # Checkpoint
                output_dir=OUTPUT_DIR,
                save_strategy="epoch",

                # Run
                do_train=True,
                do_eval=True,

                # Data processing
                num_train_epochs=args.num_train_epochs,
                per_device_train_batch_size=args.per_device_train_batch_size,
                per_device_eval_batch_size=args.per_device_eval_batch_size,
                dataloader_num_workers=args.num_workers,

                ## Optimization
                lr_scheduler_type=args.lr_scheduler_type,
                learning_rate=args.learning_rate,
                warmup_ratio=args.warmup_ratio,

                ## Regularization
                weight_decay=args.weight_decay,

                # Logging
                logging_dir='./logs',
                report_to=args.report_to,

                # Evaluation
                metric_for_best_model='auprc',
                evaluation_strategy='epoch',

                # ETC    
                load_best_model_at_end=True,
                seed=args.seed,
                fp16=args.fp16,
                )

    trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=tokenized_datasets['train'],
                # train_dataset=tokenized_datasets['train'].shard(index=1, num_shards=100), # for smoke test
                eval_dataset=tokenized_datasets['valid'],
                compute_metrics=compute_metrics,
                data_collator=data_collator,
                )

    if args.do_train:
        best_model = trainer.train()
    
    if call_wandb:
        wandb.finish()
    
    if args.do_predict:
        # TODO 여기서 Dummy Trainer 만들고, inference 함수 수정



        try:
            model = best_model
            training_args = TrainingArguments(output_dir=OUTPUT_DIR)
            model.to(device)
            trainer = Trainer(model=model,
                             args=training_args,
                             data_collator=data_collator,
                             )
        except:
            print(f"Load model from checkpoint : {args.checkpoint}")
            # Assign your checkpint directory to the --model_name_or_path argument.
            training_args = TrainingArguments(output_dir=OUTPUT_DIR,
                                              resume_from_checkpoint=args.checkpoint)
            model.to(device)
            trainer = Trainer(model=model.from_pretrained(args.checkpoint),
                             args=training_args,
                             data_collator=data_collator,
                             )

            print(f"Model from HF-hub : {model.config.name_or_path} / Model from checkpoint :  {trainer.model.config.name_or_path}") 

        pred_answer, output_prob = inference(trainer, tokenized_datasets['test'])
        pred_answer = num_to_label(pred_answer)
        output = pd.DataFrame({'id':tokenized_datasets['test']['guid'],
                               'pred_label':pred_answer,'probs':output_prob,})
        
        if (os.path.isdir('submission') == False):
            os.mkdir('submission')
        save_path = f'submission/submission-{args.wandb_run_name}.csv'
        output.to_csv(save_path, index=False)

        print(f'Submission csv file saved in {save_path}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser = add_general_args(parser, os.getcwd())
    args = parser.parse_args()
    print(args)
    # exit()

    main(args)