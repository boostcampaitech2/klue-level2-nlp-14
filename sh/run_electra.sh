python train.py \
         --epochs 10 \
         --model_name tunib/electra-ko-en-base \
         --lr 5e-5 \
         --batch_size 32 \
         --valid_batch_size 32 \
         --warmup_ratio 0.2 \
         --output_name 0929_tunib_koen_electra_base \
         --wandb_project klue_re_tunib_koelectra_taeuk \
         --run_name 0929_tunib_koen_electra_base \
         --output_dir 0929_tunib_koen_electra_base \
         --best_model_dir 0929_tunib_koen_electra_base \
         