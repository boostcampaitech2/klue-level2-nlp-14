from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import DataLoader
from load_data import *
import pandas as pd
import torch
import torch.nn.functional as F

import pickle as pickle
import numpy as np
import argparse
from tqdm import tqdm

def inference(trainer, tokenized_sent):
    """
      test dataset을 DataLoader로 만들어 준 후,
      batch_size로 나눠 model이 예측 합니다.
    """

    # dataloader = DataLoader(tokenized_sent, batch_size=32, collate_fn=collate_fn, shuffle=False)
    # model.eval()
    output_pred = []
    output_prob = []

    # # TODO 결과가 달라짐, Trainer객체 이용
    # 'token_type_ids' 처리가 누락, RoBERTa는 이에 영향을 받지 않으므로 상관 없음
    # 그러나 이를 사용하는 다른 모델의 경우 문제가 있을 수 있으며 전처리 부분을 이에 맞게 수정해야함
    # for i, data in enumerate(tqdm(dataloader)):
    #   with torch.no_grad():
    #     outputs = model(
    #         input_ids=data['input_ids'].to(device),
    #         token_type_ids=data['token_type_ids'].to(device),
    #         attention_mask=data['attention_mask'].to(device),
    #         )
    outputs = trainer.predict(tokenized_sent)

    logits = outputs[0]
    prob = F.softmax(torch.from_numpy(logits), dim=-1).detach().cpu().numpy()
    result = np.argmax(logits, axis=-1)

    output_pred.append(result)
    output_prob.append(prob)
  
    return np.concatenate(output_pred).tolist(), np.concatenate(output_prob, axis=0).tolist()

def num_to_label(label):
    """
      숫자로 되어 있던 class를 원본 문자열 라벨로 변환 합니다.
    """
    origin_label = []
    with open('data/dict_num_to_label.pkl', 'rb') as f:
        dict_num_to_label = pickle.load(f)
    for v in label:
        origin_label.append(dict_num_to_label[v])
    
    return origin_label

def load_test_dataset(dataset_dir, tokenizer):
    """
      test dataset을 불러온 후,
      tokenizing 합니다.
    """
    test_dataset = load_data(dataset_dir)
    test_label = list(map(int,test_dataset['label'].values))
    # tokenizing dataset
    tokenized_test = tokenized_dataset(test_dataset, tokenizer)
    return test_dataset['id'], tokenized_test, test_label
