export CUDA_VISIBLE_DEVICES=0

python3 run_prompt.py \
--data_dir ../datasets/klue-re \
--output_dir ../results/klue-re \
--model_type roberta \
--model_name_or_path jinmang2/roberta-large-re-tapt-20300 \
--per_gpu_train_batch_size 64 \
--gradient_accumulation_steps 1 \
--max_seq_length 139 \
--warmup_steps 0 \
--learning_rate 2e-5 \
--learning_rate_for_new_token 1e-5 \
--num_train_epochs 4 \
--weight_decay 1e-2 \
--adam_epsilon 1e-6 \
--temps temp.txt \
--wandb_run_name Test_TAPT_Aug2_roberta-large_tapt_bs64_maxlen139_focal_ws0_lr2e-05_lrt1e-05
