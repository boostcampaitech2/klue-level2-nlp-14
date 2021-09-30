# https://github.com/huggingface/transformers/blob/55695df0f7bce816b6d53ab2d43d51427ea77a75/src/transformers/integrations.py#L140
import os
import pickle
import json

import numpy as np
import torch
import random
from distutils.util import strtobool

import ray
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import PopulationBasedTraining, ASHAScheduler

from transformers import AutoConfig, \
    AutoModelForSequenceClassification, AutoTokenizer, Trainer, GlueDataset, \
    GlueDataTrainingArguments, TrainingArguments, TrainerCallback
import datasets
from datasets import Dataset, DatasetDict, Features, Value, ClassLabel

from constant import *
import preps.entity_tagging as prep
from preps.entity_tagging import *
from utils import add_general_args, inference, num_to_label, compute_metrics

# TODO Add Memory Saving Method
# class MemorySaverCallback(TrainerCallback):
#     "A callback that deleted the folder in which checkpoints are saved, to save memory"
#     def __init__(self, run_name):
#         super(MemorySaverCallback, self).__init__()
#         self.run_name = run_name

#     def on_train_begin(self, args, state, control, **kwargs):
#         print("Removing dirs...")
#         if os.path.isdir(f'./{self.run_name}'):
#             import shutil
#             shutil.rmtree(f'./{self.run_name}')
#         else:
#             print("\n\nDirectory does not exists")

def tune_transformer(num_samples=8, gpus_per_trial=0, smoke_test=False, args=None):
    def set_seeds(seed=42):
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False # for faster training, but not deterministic

    set_seeds()

    # Basic Setup
    BACKEND = args.backend
    SAVECHECKPOINT = args.save_ckpt
    METHOD = 'ASHA' # or 'PBT'
    OBJECTIVE_METRIC = 'eval_auprc' # or 'eval_f1', or 'eval_loss'

    # Directory where the hyper parameter search results(checkpoint, best hyperparameters setting) are saved 
    data_dir_name = "./hp_search" if not smoke_test else "./hp_search_test"
    data_dir = os.path.abspath(os.path.join(os.getcwd(), data_dir_name))
    if not os.path.exists(data_dir):
        os.mkdir(data_dir, 0o755)
    best_hp_dir_name = "./best_hyperparameters"
    best_hp_dir = os.path.abspath(os.path.join(data_dir, best_hp_dir_name))
    if not os.path.exists(best_hp_dir):
        os.mkdir(best_hp_dir, 0o755)

    # Optuar result dir, c.f. ray tune automatically creates the folder "ray_result"
    optuna_result_dir_name = './optuna_result'
    optuna_result_dir = os.path.abspath(os.path.join(data_dir, optuna_result_dir_name))
    if not os.path.exists(optuna_result_dir):
        os.mkdir(optuna_result_dir, 0o755)
    if not os.path.exists(optuna_result_dir):
        os.mkdir(optuna_result_dir, 0o755)

    # Directory where the initial pre-trained model saved (download from Huggingface hub) 
    cache_dir_name = './cache'
    cache_dir = os.path.abspath(os.path.join(os.getcwd(), cache_dir_name))
    if not os.path.exists(cache_dir):
        os.mkdir(cache_dir, 0o755)
    MODEL_DIR = os.path.join(cache_dir, 'models')

    model_name = args.model_name_or_path

    num_labels = len(CLASS_NAMES)


    # Setting model hyperparameter
    config =  AutoConfig.from_pretrained(model_name, num_labels=num_labels)
    config.num_labels = 30
    config.cache_dir = cache_dir
    config.id2label = IDX2LABEL
    config.label2id = LABEL2IDX


    # Download and cache tokenizer, model, and features
    print("Downloading and caching Tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    tokenizer.add_special_tokens(
    {"additional_special_tokens": list(MARKERS.values())}
    )
    

    # Triggers model saved to MODEL_DIR
    ## (to avoid the issue of the token embedding layer size)
    print(f"Downloading and caching the pre-trained model : {model_name}")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        config=config,
        cache_dir=cache_dir
    )
    
    if model.config.vocab_size < len(tokenizer):
        print("Resize the vocab size...")
        model.resize_token_embeddings(len(tokenizer))
        
    model.save_pretrained(MODEL_DIR)
    config.vocab_size = len(tokenizer)
    print(f"The pre-trained model saved in {MODEL_DIR}")

    del model

    def get_model():
        print(f"Loading the initial pre-trained model from {MODEL_DIR}")
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_DIR,
            config=config,
            cache_dir=cache_dir
        )
        if model.config.vocab_size < len(tokenizer):
            print("Resize the vocab size...")
            model.resize_token_embeddings(len(tokenizer))
        
        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda:0"
            if torch.cuda.device_count() > 1:
                net = nn.DataParallel(net)
        model.to(device)
        
        return model

    # Download data
    raw_dataset = datasets.load_dataset("jinmang2/load_klue_re", script_version="v1.0.1b", cache_dir=cache_dir)
    convert_example_to_features = partial(
    prep.convert_example_to_features,
    tokenizer=tokenizer,
    **MARKERS,
    )
    examples = raw_dataset.map(prep.mark_entity_spans)
    tokenized_datasets = examples.map(convert_example_to_features)
    max_length = 128
    data_collator = DataCollator(tokenizer, max_length=max_length)

    # Hyperparameters marked with 'config' are the targets of hyperparameter search.
    training_args = TrainingArguments(
                # Checkpoint
                output_dir="." if BACKEND == 'ray' else optuna_result_dir,
                save_strategy="epoch" if BACKEND == 'ray' or SAVECHECKPOINT == True else "no",
                save_total_limit=1,
                overwrite_output_dir=True,

                # Run
                do_train=True,
                do_eval=True,

                # Training
                num_train_epochs=1, # config
                max_steps=1 if smoke_test else -1,
                learning_rate=5e-5, # config
                per_device_train_batch_size=32, # config
                per_device_eval_batch_size=32,  # config
                ## Learning rate scheduling
                warmup_steps=0,
                ## Regularization
                weight_decay=0.01, # config

                # Logging
                logging_dir='./logs',
                report_to ="none",
                logging_strategy="steps",
                logging_steps=500,

                # Evaluation
                metric_for_best_model = 'eval_auprc',
                evaluation_strategy='epoch',
                # eval_steps = 500,
                # ETC    
                load_best_model_at_end = True if BACKEND == 'ray' else False,
                seed = 42,
                # https://stackoverflow.com/questions/68787955/cant-pickle-thread-rlock-objects-when-using-huggingface-trainer-with-ray-tun
                skip_memory_metrics=True,
                # GPU
                fp16 = True,
                no_cuda=gpus_per_trial <= 0,
#                 dataloader_num_workers=4,
                )

    # Setting Trainer, Resize the train set for the faster search
    trainer = Trainer(
        model_init=get_model,
        args=training_args,
        train_dataset=tokenized_datasets['train'].shard(index=1, num_shards=300),
        eval_dataset=tokenized_datasets['valid'].shard(index=1, num_shards=4),
        compute_metrics=compute_metrics,
        data_collator=data_collator,
#         callbacks=[MemorySaverCallback(".")]
    )

    def my_objective(metrics):
        return metrics[OBJECTIVE_METRIC]
    
    if BACKEND == 'ray':

        # Setting hyperparameter search method
        if METHOD == 'ASHA':
            def tune_config_fn(*args, **kwargs):
                return {
                        "per_device_train_batch_size": tune.choice([8, 16, 32]),
                        "per_device_eval_batch_size": 32,
                        "learning_rate": tune.choice([1e-5, 2e-5, 3e-5, 5e-5]),
                        "warmup_ratio": tune.choice([0., 0.1, 0.2, 0.6]),
                        "weihgt_decay": tune.choice([0.0, 0.01]),
                        "num_train_epochs": 1,
                        # "num_train_epochs": tune.choice([3, 4, 5, 10]),
                        "max_steps": -1
                        # "max_steps": 1 if smoke_test else -1,  # Used for smoke test.
                        }

            scheduler = ASHAScheduler(
                time_attr="training_iteration", # 
                max_t=2,
                metric=OBJECTIVE_METRIC,
                mode="max",
                reduction_factor=3,
                brackets = 1,
                )

        elif METHOD == 'PBT':
            def tune_config_fn(*args, **kwargs):
                return {
                    "per_device_train_batch_size": 32,
                    "per_device_eval_batch_size": 32,
                    "num_train_epochs": tune.choice([2, 3]),
                    "max_steps": 1 if smoke_test else -1,  # Used for smoke test.
                }

            scheduler = PopulationBasedTraining(
                time_attr="training_iteration",
                metric=OBJECTIVE_METRIC,
                mode="max",
                perturbation_interval=1,
                hyperparam_mutations={
                    "weight_decay": tune.uniform(0.0, 0.3),
                    "learning_rate": tune.uniform(1e-5, 5e-5),
                    "per_device_train_batch_size": [16, 32, 64],
                }
                )
        
        reporter = CLIReporter(
            parameter_columns={
                "weight_decay": "w_decay",
                "learning_rate": "lr",
                "per_device_train_batch_size": "train_bs/gpu",
                "num_train_epochs": "num_epochs"
            },
            metric_columns=[
                OBJECTIVE_METRIC, "eval_loss", "epoch", "training_iteration"
            ])

        best_trial = trainer.hyperparameter_search(
            hp_space=tune_config_fn,
            backend=BACKEND,
            n_trials=num_samples,
            resources_per_trial={
                "cpu": 4,
                "gpu": gpus_per_trial
            },
            scheduler=scheduler,
            keep_checkpoints_num=1,
            checkpoint_score_attr="training_iteration",
            stop={"training_iteration": 1} if smoke_test else None,
            progress_reporter=reporter,
            local_dir=os.path.join(data_dir,'ray_result'),
            name=f"tune_transformer_{METHOD}",
            log_to_file=True,
            compute_objective=my_objective
            )

    elif BACKEND == 'optuna':
        def hp_space_optuna(trial):
            # KLUE Paper Baseline
            # How to set : https://optuna.readthedocs.io/en/stable/reference/generated/optuna.trial.Trial.html?highlight=suggest_float#optuna.trial.Trial.suggest_float
            return {
                "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", [8, 16, 32]),
                "per_device_eval_batch_size": 32,
                "learning_rate": trial.suggest_categorical("learning_rate", [1e-5, 2e-5, 3e-5, 5e-5]),
                "warmup_ratio":  trial.suggest_categorical("warmup_ratio", [0., 0.1, 0.2, 0.6]),
                # "num_train_epochs":  trial.suggest_categorical("num_train_epochs",[1, 2]),
                "num_train_epochs":  trial.suggest_categorical("num_train_epochs",[3, 4, 5, 10]),
                "weight_decay":  trial.suggest_categorical("weight_decay", [0., 0.01]),
                "max_steps": 1 if smoke_test else -1
            }

        best_trial = trainer.hyperparameter_search(
            hp_space=hp_space_optuna,
            backend=BACKEND,
            n_trials=num_samples,
            compute_objective=my_objective
            )

    
    print(best_trial)
    trial_save_path = os.path.join(best_hp_dir,
        f'{BACKEND}_{model_name.replace('/','-')}_{best_trial.run_id}_{OBJECTIVE_METRIC}_{best_trial.objective}.json')
    with open(trial_save_path, 'w', encoding='utf-8') as f:\
            json.dump(best_trial.hyperparameters, f, ensure_ascii=False, indent=4)
    print(f"Save best trial file in {trial_save_path}")

if __name__ == "__main__":
    '''
    Usage example
    python hps.py --model_name_or_path klue/roberta-large
    # checkpoint 및 best hyperparameter는 `hp_search` 폴더 내에 저장됨
    # best hyperparameter는 json 형태의 파일로 저장
    # 용량에 주의! num_samples 만큼의 checkpoint를 저장
    - 저장하지 않도록 설정하려면 --save_ckpt False(기본값)
    - --save_chkp True로 설정한 경우, optuna는 가장 마지막 체크포인트만 저장되고,
      ray는 무시하고 저장됨
    '''
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--smoke-test", action="store_true", help="Finish quickly for testing")
    parser.add_argument(
        "--ray-address",
        type=str,
        default=None,
        help="Address to use for Ray. "
        "Use \"auto\" for cluster. "
        "Defaults to None for local.")
    parser.add_argument(
        "--server-address",
        type=str,
        default=None,
        required=False,
        help="The address of server to connect to if using "
        "Ray Client.")

    # Pre-trained model name : e.g. "klue/roberta-small"
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        required=True,
        help="model identifier from huggingface.co/models",
    )
    parser.add_argument("--save_ckpt",
        default=False,
        type=lambda x: bool(strtobool(x)),
        help="do predict"
    )

    parser.add_argument(
        "--backend",
        default="ray",
        type=str,
        help="option: [ray, optuna]",
    )

    parser.add_argument(
        "--num_samples",
        default=2,
        type=int,
        help="number of random search trials",
    )

    args, _ = parser.parse_known_args()
    if args.backend=='ray':
        ray.init()
        # ray.init(local_mode=True)
    # if args.smoke_test:
    #     tune_transformer(num_samples=1, gpus_per_trial=1, smoke_test=True, args=args)
        # You can change the number of GPUs here:
    tune_transformer(num_samples=args.num_samples, gpus_per_trial=1, args=args)