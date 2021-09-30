#!/bin/bash

set +x

CACHE_DIR="cache"
RUN_NAME="baseline-0929-roberta-large-warmup0.2_lr3e-5"
OUTPUT_DIR="klue_dir"
DATA_DIR="data"
VERSION="v1.0.1b"
WANDB_PROJECT="klue_re_klue-roberta-large-daeug"

# Example setting
python run.py --output_dir ${OUTPUT_DIR}/${RUN_NAME} \
                --model_name_or_path "klue/roberta-large" \
                --checkpoint "/opt/ml/klue_re/klue_dir/klue-roberta-large-warmup0.2_3e-5/checkpoint-2436" \
                --dataset_ver ${VERSION} \
                --cache_dir ${CACHE_DIR} \
                --max_seq_length 128 \
                --seed 42 \
                --do_train False \
                --do_eval False \
                --do_predict True \
                --num_train_epochs 4 \
                --per_device_train_batch_size 32 \
                --per_device_eval_batch_size 32 \
                --num_workers 8 \
                --learning_rate 3e-5 \
                --lr_scheduler_type "linear" \
                --warmup_ratio 0.2 \
                --weight_decay 0.01 \
                --report_to "none" \
                --wandb_project ${WANDB_PROJECT} \
                --wandb_run_name ${RUN_NAME} \
                --metric_key "eval_micro_f1" \
                --fp16 True \
                --gpus 0