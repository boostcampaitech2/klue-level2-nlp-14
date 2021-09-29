# klue_re
KLUE Releation Extraction Tasks

## Directory
![image](https://user-images.githubusercontent.com/41335296/135213857-c0d17681-501b-44b1-881f-21471c45bb41.png)

## Usage

### Training
1. Set up your hyperparameter by edditing the `./run.sh` file.
- Training & Evaluation: `--do_train True --do_eval True`
- Submission: If you set `--do_predict True`, then `./submission/submission-{RUN_NAME}.csv` file will be created.

    ```bash
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
                    --do_train True \
                    --do_eval True \
                    --do_predict False \
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
    ```

2. Run shell script
    ```bash
    $ bash run.sh
    ```

### Hyperparameter Search
1. Run `hps.py` file
- Ray Tune `AsyncHyperBandScheduler`(default) or `PopulationBasedTraining`
- The default search space is from KLUE paper
- To change Scheduler or objective metric, fix this line.
    ```python
        # Basic Setup
        METHOD = 'ASHA' # or 'PBT'
        OBJECTIVE_METRIC = 'eval_auprc' # or 'eval_f1', or 'eval_loss'

    ```
- To modify the search space, fix this line.
    ```python
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
    ```

```bash
$ python hps.py --model_name_or_path klue/roberta-large
```

## Reference
- [Ray Tune - pbt_transformers_example](https://docs.ray.io/en/master/tune/examples/pbt_transformers.html)

```
@misc{park2021klue,
      title={KLUE: Korean Language Understanding Evaluation}, 
      author={Sungjoon Park and Jihyung Moon and Sungdong Kim and Won Ik Cho and Jiyoon Han and Jangwon Park and Chisung Song and Junseong Kim and Yongsook Song and Taehwan Oh and Joohong Lee and Juhyun Oh and Sungwon Lyu and Younghoon Jeong and Inkwon Lee and Sangwoo Seo and Dongjun Lee and Hyunwoo Kim and Myeonghwa Lee and Seongbo Jang and Seungwon Do and Sunkyoung Kim and Kyungtae Lim and Jongwon Lee and Kyumin Park and Jamin Shin and Seonghyun Kim and Lucy Park and Alice Oh and Jung-Woo Ha and Kyunghyun Cho},
      year={2021},
      eprint={2105.09680},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
