import os
from glob import glob
import argparse

import numpy as np
import pandas as pd

from solution.args import (
    HfArgumentParser,
    DataArguments,
    NewTrainingArguments,
    ModelingArguments,
    ProjectArguments,
)
from solution.utils import (
    softmax,
)


def inference(command_args):
    label2num = {'no_relation': 0, 'org:top_members/employees': 1, 'org:members': 2, 'org:product': 3, 'per:title': 4, 'org:alternate_names': 5, 'per:employee_of': 6, 'org:place_of_headquarters': 7, 'per:product': 8, 'org:number_of_employees/members': 9, 'per:children': 10, 'per:place_of_residence': 11, 'per:alternate_names': 12, 'per:other_family': 13, 'per:colleagues': 14, 'per:origin': 15, 'per:siblings': 16, 'per:spouse': 17, 'org:founded': 18, 'org:political/religious_affiliation': 19, 'org:member_of': 20, 'per:parents': 21, 'org:dissolved': 22, 'per:schools_attended': 23, 'per:date_of_death': 24, 'per:date_of_birth': 25, 'per:place_of_birth': 26, 'per:place_of_death': 27, 'org:founded_by': 28, 'per:religion': 29}
    num2label = {0: 'no_relation', 1: 'org:top_members/employees', 2: 'org:members', 3: 'org:product', 4: 'per:title', 5: 'org:alternate_names', 6: 'per:employee_of', 7: 'org:place_of_headquarters', 8: 'per:product', 9: 'org:number_of_employees/members', 10: 'per:children', 11: 'per:place_of_residence', 12: 'per:alternate_names', 13: 'per:other_family', 14: 'per:colleagues', 15: 'per:origin', 16: 'per:siblings', 17: 'per:spouse', 18: 'org:founded', 19: 'org:political/religious_affiliation', 20: 'org:member_of', 21: 'per:parents', 22: 'org:dissolved', 23: 'per:schools_attended', 24: 'per:date_of_death', 25: 'per:date_of_birth', 26: 'per:place_of_birth', 27: 'per:place_of_death', 28: 'org:founded_by', 29: 'per:religion'}

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
    
    run_name = training_args.run_name
    
    df_list = sorted(glob(f"prediction/submission_{run_name}_fold*.csv"), key=lambda x: x.split('-')[-1])
    df_list = list(map(pd.read_csv, df_list))
    
    df_probs = list(map(lambda x : np.stack(x['probs'].apply(eval).values,0), df_list))
    df_probs = np.stack(df_probs, 1)
    
    df_soft_voting_prob = softmax(df_probs.sum(1), axis=1)
    
    pd.set_option('mode.chained_assignment',  None) # SettingWithCopyWarning 경고 무시하기
    
    df_submit = df_list[0].copy()
    for i in range(len(df_soft_voting_prob)):
        df_submit['probs'][i] = list(df_soft_voting_prob[i])
    df_submit['pred_label'] = df_soft_voting_prob.argmax(-1)
    df_submit['pred_label'] = df_submit['pred_label'].map(num2label)
    
    for i in range(5):
        print(f"fold{i+1}과의 결과 유사도 : {(df_submit['pred_label'] == df_list[i]['pred_label']).sum() / len(df_submit)}")
    
    df_submit.to_csv(f'prediction/submission_{run_name}_fold_complete.csv', index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default="config/kfold.yaml", help='config file path (default: config/kfold.yaml)')
    command_args = parser.parse_args()
    print(command_args)

    inference(command_args)
