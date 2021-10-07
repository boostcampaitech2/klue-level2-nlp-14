# https://github.com/huggingface/transformers/blob/55695df0f7bce816b6d53ab2d43d51427ea77a75/src/transformers/integrations.py#L140
import os
import sys
import json

from functools import partial

import torch

from datasets import load_dataset, concatenate_datasets

from transformers import AutoConfig, \
    AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments

from solution.args import (
    HfArgumentParser,
    DataArguments,
    NewTrainingArguments,
    ModelingArguments,
    ProjectArguments,
    HPSearchArguments
)
from solution.data import (
    COLLATOR_MAP,
    mark_entity_spans as _mark_entity_spans,
    convert_example_to_features as _convert_example_to_features,
)
from solution.trainers import (
    TRAINER_MAP,
)
from solution.utils import (
    set_seeds,
    TASK_METRIC_MAP,
    TASK_INFOS_MAP,
)
import solution.models as models

import ray
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import PopulationBasedTraining, ASHAScheduler

def tune_transformer(num_samples=8, gpus_per_trial=0, smoke_test=False, args=None):

    data_args, training_args, model_args, project_args, hps_args = args
    
    # Set seed
    set_seeds(training_args.seed)
    
    task_infos = TASK_INFOS_MAP[project_args.task]
    compute_metrics = TASK_METRIC_MAP[project_args.task]

    # Basic Setup
    # Directory where the hyper parameter search results(checkpoint, best hyperparameters setting) are saved 
    data_dir_name = "./hp_search" if not smoke_test else "./hp_search_test"
    data_dir = os.path.abspath(os.path.join(os.getcwd(), data_dir_name))
    if not os.path.exists(data_dir):
        os.mkdir(data_dir, 0o755)
    best_hp_dir_name = "./best_hyperparameters"
    best_hp_dir = os.path.abspath(os.path.join(data_dir, best_hp_dir_name))
    if not os.path.exists(best_hp_dir):
        os.mkdir(best_hp_dir, 0o755)

    # Optuna result dir, c.f. ray tune automatically creates the folder "ray_result"
    optuna_result_dir_name = './optuna_result'
    optuna_result_dir = os.path.abspath(os.path.join(data_dir, optuna_result_dir_name))
    if not os.path.exists(optuna_result_dir):
        os.mkdir(optuna_result_dir, 0o755)
    if not os.path.exists(optuna_result_dir):
        os.mkdir(optuna_result_dir, 0o755)

    # Directory where the initial pre-trained model saved (download from Huggingface hub) 
    cache_dir_name = './' + data_args.data_cache_dir
    cache_dir = os.path.abspath(os.path.join(os.getcwd(), cache_dir_name))
    cache_dir = os.path.abspath(cache_dir_name)
    if not os.path.exists(cache_dir):
        os.mkdir(cache_dir, 0o755)
    MODEL_DIR = os.path.join(cache_dir, 'models')

    model_name = model_args.model_name_or_path

    # Get training data
    train = load_dataset("jinmang2/load_klue_re", script_version="v3.0.0b", split='train')
    aug1  = load_dataset("jinmang2/load_klue_re", script_version="v3.0.0b", split='aug1')
    valid = load_dataset("jinmang2/load_klue_re", script_version="v3.0.0b", split='valid')
    dataset = concatenate_datasets([train, aug1])
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    tokenizer.add_special_tokens(
        {"additional_special_tokens": list(task_infos.markers.values())}
    )
    collate_cls = COLLATOR_MAP[data_args.collator_name]
    data_collator = collate_cls(tokenizer)
    
    # Preprocess and tokenizing
    mark_entity_spans = partial(_mark_entity_spans, **task_infos.markers)
    convert_example_to_features = partial(
        _convert_example_to_features,
        tokenizer=tokenizer,
        **task_infos.markers,
    )
    
    examples = dataset.map(mark_entity_spans)
    tokenized_datasets = examples.map(convert_example_to_features)
    valid_examples = valid.map(mark_entity_spans)
    valid_tokenized_datasets = valid_examples.map(convert_example_to_features)
    
    # remove unused feature names
    #features_name = list(tokenized_datasets.features.keys())
    #features_name.pop(features_name.index("input_ids"))
    #features_name.pop(features_name.index("label"))
    #tokenized_datasets = tokenized_datasets.remove_columns(features_name)

    train_dataset = tokenized_datasets
    eval_dataset = valid_tokenized_datasets
    #if training_args.do_eval:
    #    try:
    #        eval_dataset = tokenized_datasets["valid"]
    #    except KeyError:
    #        print("Dataset Version Error")
    #        return None

    # Triggers model saved to MODEL_DIR
    ## (to avoid the issue of the token embedding layer size)
    print(f"Downloading and caching the pre-trained model : {model_name}")
    config = AutoConfig.from_pretrained(
            model_args.model_name_or_path,
            num_labels=task_infos.num_labels,
            cache_dir=model_args.model_cache_dir,
            id2label=task_infos.id2label,
            label2id=task_infos.label2id,
        )

    model_cls = getattr(models, model_args.architectures,
                            AutoModelForSequenceClassification)
    model = model_cls.from_pretrained(
            model_args.model_name_or_path,
            config=config,
            cache_dir=model_args.model_cache_dir,
    )
    
    if model.config.vocab_size < len(tokenizer):
        print("Resize the vocab size...")
        model.resize_token_embeddings(len(tokenizer))
        
    model.save_pretrained(MODEL_DIR)
    config.vocab_size = len(tokenizer)
    print(f"The pre-trained model saved in {MODEL_DIR}")

    del model

    # Get model
    def model_init():
        checkpoint = MODEL_DIR
        # Load Checkpoint
        model = model_cls.from_pretrained(
            checkpoint,
            config=config,
            cache_dir=model_args.model_cache_dir,
        )
        if model.config.vocab_size < len(tokenizer):
            print("Resize the vocab size...")
            model.resize_token_embeddings(len(tokenizer))

        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda:0"
            if torch.cuda.device_count() > 1:
                model = torch.nn.DataParallel(model)
        model.to(device)

        return model

    print("hps_args.save_ckpt",hps_args.save_ckpt)
    training_args.output_dir = "." if hps_args.backend == 'ray' else optuna_result_dir
    training_args.save_strategy = "epoch" if hps_args.backend == 'ray' or hps_args.save_ckpt else "no"
    training_args.max_steps = 1 if smoke_test else -1
    training_args.metric_for_best_model = hps_args.objective_metric
    training_args.load_best_model_at_end = True if hps_args.backend == 'ray' else False

    # https://stackoverflow.com/questions/68787955/cant-pickle-thread-rlock-objects-when-using-huggingface-trainer-with-ray-tun

    # Setting Trainer, Resize the train set for the faster search
    trainer_class = TRAINER_MAP[training_args.trainer_class]
    trainer = trainer_class(
        args=training_args,
        model_init=model_init,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # Hyperparameter Search Settings
    def my_objective(metrics):
        return metrics[hps_args.objective_metric]
    
    if hps_args.backend == 'ray':
        # Setting hyperparameter search method
        if hps_args.method == 'ASHA':
            def tune_config_fn(*args, **kwargs):
                return {
                        "per_device_train_batch_size": tune.choice(hps_args.hp_per_device_train_batch_size),
                        "per_device_eval_batch_size": hps_args.hp_per_device_eval_batch_size,
                        "learning_rate": tune.choice(hps_args.hp_learning_rate),
                        "warmup_ratio": tune.choice(hps_args.hp_warmup_ratio),
                        "weihgt_decay": tune.choice(hps_args.hp_weight_decay),
                        "num_train_epochs": 1 if hps_args.smoke_test else tune.choice(hps_args.hp_num_train_epochs),
                        "max_steps": 1 if smoke_test else -1
                        }

            scheduler = ASHAScheduler(
                time_attr="training_iteration", # 
                max_t=2,
                metric=hps_args.objective_metric,
                mode="max",
                reduction_factor=3,
                brackets = 1,
                )

        elif hps_args.method == 'PBT':
            def tune_config_fn(*args, **kwargs):
                return {
                    "per_device_train_batch_size": hps_args.hp_per_device_train_batch_size,
                    "per_device_eval_batch_size": hps_args.hp_per_device_eval_batch_size,
                    "num_train_epochs": tune.choice(hps_args.hp_num_train_epochs),
                    "max_steps": 1 if hps_args.smoke_test else -1,  # Used for smoke test.
                }

            scheduler = PopulationBasedTraining(
                time_attr="training_iteration",
                metric=hps_args.objective_metric,
                mode="max",
                perturbation_interval=1,
                hyperparam_mutations={
                    "weight_decay": tune.uniform(hps_args.hp_weight_decay[0], hps_args.hp_weight_decay[-1]),
                    "learning_rate": tune.uniform(hps_args.hp_weight_decay[0], hps_args.hp_weight_decay[-1]),
                    "per_device_train_batch_size": hps_args.hp_per_device_train_batch_size,
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
                hps_args.objective_metric, "eval_loss", "epoch", "training_iteration"
            ])

        best_trial = trainer.hyperparameter_search(
            hp_space=tune_config_fn,
            backend=hps_args.backend,
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
            name=f"tune_transformer_{hps_args.method}",
            log_to_file=True,
            compute_objective=my_objective
            )

    elif hps_args.backend == 'optuna':
        def hp_space_optuna(trial):
            # KLUE Paper Baseline
            # How to set : https://optuna.readthedocs.io/en/stable/reference/generated/optuna.trial.Trial.html?highlight=suggest_float#optuna.trial.Trial.suggest_float
            return {
                "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", hps_args.hp_per_device_train_batch_size),
                "per_device_eval_batch_size": 32,
                "learning_rate": trial.suggest_categorical("learning_rate", hps_args.hp_learning_rate),
                "warmup_ratio":  trial.suggest_categorical("warmup_ratio", hps_args.hp_warmup_ratio),
                "weight_decay":  trial.suggest_categorical("weight_decay", hps_args.hp_weight_decay),
                "num_train_epochs":  trial.suggest_categorical("num_train_epochs", hps_args.hp_num_train_epochs),
                "max_steps": 1 if smoke_test else -1
            }

        best_trial = trainer.hyperparameter_search(
            hp_space=hp_space_optuna,
            backend=hps_args.backend,
            n_trials=num_samples,
            compute_objective=my_objective
            )

    print(best_trial)
    model_name_or_path = model_args.model_name_or_path.replace('/','-')
    trial_save_path = os.path.join(best_hp_dir,
        f'{hps_args.backend}_{model_name_or_path}_{best_trial.run_id}_{hps_args.objective_metric}_{best_trial.objective}.json')
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
    parser = HfArgumentParser(
        (DataArguments,
         NewTrainingArguments,
         ModelingArguments,
         ProjectArguments,
         HPSearchArguments)
    )

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # read args from json file
        args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    elif len(sys.argv) == 2 and sys.argv[1].endswith(".yaml"):
        args = parser.parse_yaml_file(yaml_file=os.path.abspath(sys.argv[1]))
    else:
        # read args from shell script or real arguments
        args = parser.parse_args_into_dataclasses()

    _, _, _, _, hps_args = args

    if hps_args.backend=='ray':
        ray.init()
        # ray.init(local_mode=True)
    if hps_args.smoke_test:
        tune_transformer(num_samples=2,
                         gpus_per_trial=1,
                         smoke_test=hps_args.smoke_test,
                         args=args)
    else:
        tune_transformer(num_samples=hps_args.num_samples,
                        gpus_per_trial=1,
                        args=args)