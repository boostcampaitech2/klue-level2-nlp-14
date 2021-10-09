import os
import sys
import argparse

from functools import partial
from typing import Tuple, List, Any, Dict
from tqdm import tqdm

import numpy as np
import pandas as pd

import torch
import torch.nn as nn

from datasets import load_dataset, concatenate_datasets

import transformers
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForSequenceClassification,
)
from transformers import TrainingArguments, Trainer
from transformers.configuration_utils import PretrainedConfig

from solution.args import (
    HfArgumentParser,
    DataArguments,
    NewTrainingArguments,
    ModelingArguments,
    ProjectArguments,
)
from solution.data import (
    COLLATOR_MAP,
    PREPROCESSING_PIPELINE,
    kfold_split,

from solution.models import (
    MODEL_INIT_FUNC,
)
from solution.trainers import (
    TRAINER_MAP,
)
from solution.utils import (
    softmax,
    set_seeds,
    get_confusion_matrix,
    TASK_METRIC_MAP,
    TASK_INFOS_MAP,
    CONFIG_FILE_NAME,
    PYTORCH_MODEL_NAME,
    INFERENCE_PIPELINE,
)


# debug for cuda
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def main(command_args):
    parser = HfArgumentParser(
        (DataArguments,
         NewTrainingArguments,
         ModelingArguments,
         ProjectArguments,)
    )
    if command_args.config.endswith(".json"):
        # read args from json file
        args = parser.parse_json_file(json_file=os.path.abspath(command_args.config))
    elif command_args.config.endswith(".yaml"):
        args = parser.parse_yaml_file(yaml_file=os.path.abspath(command_args.config))
    else:
        # read args from shell script or real arguments
        args = parser.parse_args_into_dataclasses()

    data_args, training_args, model_args, project_args = args

    # Set seed
    set_seeds(training_args.seed)

    checkpoint = project_args.checkpoint

    task_infos = TASK_INFOS_MAP[project_args.task]
    compute_metrics = TASK_METRIC_MAP[project_args.task]

    # Get training data
    dataset = load_dataset(
        data_args.name,
        script_version=data_args.revision,
        cache_dir=data_args.data_cache_dir,
    )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    tokenizer.add_special_tokens(
        {"additional_special_tokens": list(task_infos.markers.values())}
    )
    collate_cls = COLLATOR_MAP[data_args.collator_name]
    prep_pipeline = PREPROCESSING_PIPELINE[data_args.prep_pipeline_name]
    data_collator = collate_cls(tokenizer, max_length=data_args.max_length)

    # Preprocess and tokenizing
    tokenized_datasets = prep_pipeline(dataset,
                                       tokenizer,
                                       task_infos,)

    # =============================================================
    # UnitTest input
    # print(tokenized_datasets["train"][0])
    # return None
    # =============================================================

    # Get model
    _model_init = MODEL_INIT_FUNC[model_args.model_init]
    model_init = partial(_model_init,
                         model_args=model_args,
                         task_infos=task_infos,
                         tokenizer=tokenizer,)

    # =============================================================
    # UnitTest RECENT
    # model = model_init()
    # output = model(torch.LongTensor(2, 10).random_(10000),
    #                head_ids=torch.LongTensor([0,8]),
    #                labels=torch.LongTensor(
    #                    [[0,0,4,0,0,2,0,0,0,0,0,0,0,],
    #                     [0,8,0,0,0,0,0,0,0,0,1,0,0,]]))
    # return output
    # =============================================================

    # Set-up WANDB
    os.environ["WANDB_PROJECT"] = project_args.wandb_project

    call_wandb = True
    try:
        os.environ["WANDB_PROJECT"]

    except KeyError:
        call_wandb = False

    if call_wandb:
        import wandb
        wandb.login()

    # augmentation dataset 선택 및 결합
    if data_args.augment != 'original':
        train_dataset = concatenate_datasets([tokenized_datasets['train'], tokenized_datasets[data_args.augment]])
    else:
        train_dataset = tokenized_datasets["train"]

    # TODO datasetdict가 아닌 경우 처리
    try:
        eval_dataset = tokenized_datasets["valid"]
    except KeyError:
        eval_dataset = None

    if project_args == "tapt":
        test_dataset = load_dataset(data_args.name,
                                    script_version=data_args.revision,
                                    cache_dir=data_args.data_cache_dir,
                                    split="test",)
        test_dataset = prep_pipeline(test_dataset,
                                     tokenizer,
                                     task_infos,
                                     mode="train",)
        train_dataset = concatenate_datasets([train_dataset, test_dataset])
        eval_dataset = None

    # train/valid split
    if command_args.fold > 0:
        # kfold
        train_dataset, eval_dataset = kfold_split(train_dataset, n_splits=5, fold=command_args.fold, random_state=training_args.seed)
        training_args.run_name = training_args.run_name + f"_fold{command_args.fold}"
        training_args.output_dir = training_args.output_dir + f"/fold{command_args.fold}"
        project_args.save_model_dir = project_args.save_model_dir + f"/fold{command_args.fold}"

    # =============================================================
    # Smoke test
    # train_dataset = train_dataset.select([i for i in range(1000)])
    # if eval_dataset is not None:
    #     eval_dataset = eval_dataset.select([i for i in range(500)])
    # =============================================================

    trainer_class = TRAINER_MAP[training_args.trainer_class]
    trainer = trainer_class(
        model_init=model_init,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # Training
    if training_args.do_train:
        trainer.train()
        checkpoint = project_args.save_model_dir
        trainer.model.save_pretrained(checkpoint)
        tokenizer.save_pretrained(checkpoint)

    if training_args.do_predict and not project_args.task == "tapt":
        # Load Checkpoint
        ckpt_config_file = os.path.join(checkpoint, CONFIG_FILE_NAME)
        ckpt_model_file = os.path.join(checkpoint, PYTORCH_MODEL_NAME)
        config = PretrainedConfig.from_json_file(ckpt_config_file)
        # We load the model state dict on the CPU to avoid an OOM error.
        state_dict = torch.load(ckpt_model_file, map_location="cpu")
        # If the model is on the GPU, it still works!
        trainer._load_state_dict_in_model(state_dict)
        del state_dict

        # Inference
        test_dataset = load_dataset(
            data_args.name,
            script_version=data_args.revision,
            cache_dir=data_args.data_cache_dir,
            split="test",
        )
        test_id = test_dataset["guid"]
        tokenized_test_datasets = prep_pipeline(test_dataset,
                                                tokenizer,
                                                task_infos,
                                                mode="test",)

        infer_pipeline = INFERENCE_PIPELINE[project_args.infer_pipeline_name]
        probs, pred_answer = infer_pipeline(tokenized_test_datasets,
                                            trainer=trainer,
                                            task_infos=task_infos,
                                            training_args=training_args,)
        output = pd.DataFrame(
            {
                'id':test_id,
                'pred_label':pred_answer,
                'probs':probs,
            }
        )
        submir_dir = training_args.output_dir
        run_name = training_args.run_name
        output.to_csv(f'./{submir_dir}/submission_{run_name}.csv', index=False)

    if project_args.do_analysis:

        del trainer
        torch.cuda.empty_cache()

        # Load & Preprocess the Dataset(for all samples of train dataset)
        train_dataset = load_dataset(
            data_args.name,
            script_version=data_args.revision,
            cache_dir=data_args.data_cache_dir,
            split="train",
        )
        train_dataset = train_dataset
        train_id = train_dataset["guid"]
        tokenized_train_datasets = pipeline(train_dataset,
                                           tokenizer,
                                           task_infos,)

        train_dataloader = torch.utils.data.DataLoader(tokenized_train_datasets,
                                      collate_fn=data_collator,
                                      batch_size=training_args.per_device_train_batch_size, shuffle=False,
                                      num_workers=4, pin_memory=True)

        model = AutoModelForSequenceClassification.from_pretrained(checkpoint, output_hidden_states=True)
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        model.to(device)

        # Path Setup : analysis results will be saved in `./analysis`
        submir_dir = project_args.submit_dir
        run_name = training_args.run_name
        analysis_dir = f'./analysis/{run_name}'
        if (os.path.isdir(analysis_dir) == False):
            os.makedirs(analysis_dir)

        # Inference
        output_logit = []
        output_prob = []
        output_pred = []

        batch_size=training_args.per_device_train_batch_size
        embeddings = np.zeros((len(tokenized_train_datasets), model.config.hidden_size), dtype=np.float32)

        for i, data in enumerate(tqdm(train_dataloader)):

            with torch.no_grad():
                outputs = model(
                        input_ids=data['input_ids'].to(device),
                        attention_mask=data['attention_mask'].to(device),
                        token_type_ids=data['token_type_ids'].to(device) if 'token_type_ids' in data.keys() else None,
                        )

            hidden_state = outputs[1][-1][:, 0, :].detach().cpu().numpy()
            if len(hidden_state) == batch_size:
                embeddings[i*batch_size:(i+1)*batch_size] = hidden_state
            else:
                embeddings[i*batch_size:i*batch_size + len(hidden_state)] = hidden_state

            logits = outputs[0]
            prob = nn.functional.softmax(logits, dim=-1).detach().cpu().numpy()
            logits = logits.detach().cpu().numpy()
            result = np.argmax(logits, axis=-1)

            output_logit.append(logits)
            output_pred.append(result)
            output_prob.append(prob)

        # Save Embeddings
        np.save(os.path.join(analysis_dir, f'embeddings.npy'), embeddings)
        print('Embedding vectors saved in ' , os.path.join(analysis_dir, f'embeddings.npy'))

        # Save Confusion Matrix & Dataframe
        pred_answer = np.concatenate(output_pred).tolist()
        output_prob = np.concatenate(output_prob, axis=0).tolist()
        output = pd.DataFrame({'id':train_id, 'pred_label':pred_answer,'probs':output_prob,})

        cm_fig = get_confusion_matrix(output['pred_label'].values,
                                      tokenized_train_datasets['label'],
                                      is_logit=False)

        cm_fig.savefig(os.path.join(analysis_dir, f'confusion_mtx.png'), dpi=300)
        torch.save(output, os.path.join(analysis_dir, 'data_frame.pt'))
        print('Dataframe & Confusion matrix saved in ' , analysis_dir)
    print('---- Finish! ----')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--fold', type=int, default=0, help='k-fold fold num: 1~5 & no k-fold: 0 (default)')
    parser.add_argument('--config', type=str, default="config/kfold.yaml", help='config file path (default: config/kfold.yaml)')
    command_args = parser.parse_args()

    main(command_args)
