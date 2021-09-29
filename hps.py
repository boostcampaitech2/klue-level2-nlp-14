import os
import pickle

import numpy as np
import torch
import random

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
        torch.manual_seed(seed)``
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False # for faster training, but not deterministic

    set_seeds()

    # Basic Setup
    METHOD = 'ASHA' # or 'PBT'
    OBJECTIVE_METRIC = 'eval_auprc' # or 'eval_f1', or 'eval_loss'

    data_dir_name = "./hp_search" if not smoke_test else "./hp_search_test"
    data_dir = os.path.abspath(os.path.join(os.getcwd(), data_dir_name))
    
    # Directory where the initial pre-trained model saved (download from Huggingface hub)

    if not os.path.exists(data_dir):
        os.mkdir(data_dir, 0o755)

    cache_dir_name = './cache'
    cache_dir = os.path.abspath(os.path.join(os.getcwd(), cache_dir_name))
    if not os.path.exists(cache_dir):
        os.mkdir(cache_dir, 0o755)

    MODEL_DIR = os.path.join(cache_dir, 'models')
    
    model_name = args.model_name_or_path
    task_name = "re"
    task_data_dir = os.path.join(data_dir, task_name.upper())

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

    # 
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
                output_dir=".",
                save_strategy="epoch",
                # Run
                do_train=True,
                do_eval=True,
                # Training
                num_train_epochs=1,            
                max_steps=-1,
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
                # Evaluation
                metric_for_best_model = 'eval_auprc',
                evaluation_strategy='epoch',
                eval_steps = 500,           
                # ETC    
                load_best_model_at_end = True,
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
        train_dataset=tokenized_datasets['train'].shard(index=1, num_shards=5),
        eval_dataset=tokenized_datasets['valid'],
        compute_metrics=compute_metrics,
        data_collator=data_collator,
#         callbacks=[MemorySaverCallback(".")]
    )
    
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
                    "max_steps": 1 if smoke_test else -1,  # Used for smoke test.
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
        
    def my_objective(metrics):
        return metrics[OBJECTIVE_METRIC]

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
        backend="ray",
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
        local_dir="~/ray_results/",
        name="tune_transformer_pbt",
        log_to_file=True,
        compute_objective=my_objective
        )
    
    print(best_trial)
    torch.save(best_trial, os.path.join(data_dir, f"{model_name.replace('/', '-')}-hps_best_trial.pt"))
    print(f"Save best trial file in {data_dir}")


if __name__ == "__main__":
    '''
    Usage example
    python hps.py --model_name_or_path klue/roberta-large
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

    args, _ = parser.parse_known_args()

    if args.smoke_test:
        ray.init(local_mode=True)
    elif args.server_address:
        ray.init(f"ray://{args.server_address}")
    else:
        ray.init(args.ray_address)

    if args.smoke_test:
        tune_transformer(num_samples=1, gpus_per_trial=1, smoke_test=True, args=args)
    else:
        # You can change the number of GPUs here:
        tune_transformer(num_samples=2, gpus_per_trial=1, args=args)