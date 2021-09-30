python train.py \
         --epochs 10 \
         --model_name tunib/electra-ko-en-base \
         --lr 3e-5 \
         --batch_size 32 \
         --valid_batch_size 32 \
         --warmup_ratio 0.2 \
         --output_name 0929_tunib_koen_electra_base_3e5 \
         --wandb_project klue_re_tunib_koelectra_taeuk \
         --run_name 0929_tunib_koen_electra_base_3e5 \
         --output_dir 0929_tunib_koen_electra_base_3e5 \
         --best_model_dir 0929_tunib_koen_electra_base_3e5 \
         