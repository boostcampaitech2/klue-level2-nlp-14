#!/bin/bash

set +x

CONFIG_DIR="configs/kfold.yaml"
KFOLD=$(seq 1 5)

# kfold setting
for fold in $KFOLD
do
    python new_run.py --config ${CONFIG_DIR} --fold ${fold}
done

python inference_kfold.py --config ${CONFIG_DIR}