import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from arguments import get_args_parser

RELATION_CLASS = [
    'no_relation',
    'org:top_members/employees',
    'org:members',
    'org:product',
    'per:title',
    'org:alternate_names',
    'per:employee_of',
    'org:place_of_headquarters',
    'per:product',
    'org:number_of_employees/members',
    'per:children',
    'per:place_of_residence',
    'per:alternate_names',
    'per:other_family',
    'per:colleagues',
    'per:origin',
    'per:siblings',
    'per:spouse',
    'org:founded',
    'org:political/religious_affiliation',
    'org:member_of',
    'per:parents',
    'org:dissolved',
    'per:schools_attended',
    'per:date_of_death',
    'per:date_of_birth',
    'per:place_of_birth',
    'per:place_of_death',
    'org:founded_by',
    'per:religion'
]

NUM2LABEL = {k:v for k, v in enumerate(RELATION_CLASS)}
LABEL2NUM = {v:k for k, v in NUM2LABEL.items()}


def softmax(arr: np.ndarray, axis: int = -1):
    c = arr.max(axis=axis, keepdims=True)
    s = arr - c
    nominator = np.exp(s)
    denominator = nominator.sum(axis=axis, keepdims=True)
    probs = nominator / denominator
    return probs

def get_temps(tokenizer):
    args = get_args_parser()
    temps = {}
    with open(args.data_dir + "/" + args.temps, "r") as f:
        for i in f.readlines():
            i = i.strip().split("\t")
            info = {}
            
            info['name'] = i[1].strip()
            info['temp'] = [
                    [tokenizer.mask_token, ':'],
                    # ['the', tokenizer.mask_token],
                    [tokenizer.mask_token, tokenizer.mask_token, tokenizer.mask_token], 
                    [tokenizer.mask_token, ':'],
                    # ['the', tokenizer.mask_token],
             ]
            print (i)
            info['labels'] = [
                (i[2],),
                (i[3],i[4],i[5]),
                (i[6],)
            ]
            print (info)
            temps[info['name']] = info
    return temps


def set_seeds(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False # for faster training, but not deterministic


class FocalLoss(torch.nn.Module):
    def __init__(self, alpha: float = None, gamma: float = 0.5, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float,int)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)

        logpt = F.log_softmax(input)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()


def get_confusion_matrix(logit_or_preds, labels, is_logit=True):
    preds = np.argmax(logit_or_preds, axis=1).ravel() if is_logit else logit_or_preds
    cm = confusion_matrix(labels, preds)
    norm_cm = cm / np.sum(cm, axis=1)[:,None]
    cm = pd.DataFrame(norm_cm, index=RELATION_CLASS, columns=RELATION_CLASS)
    fig = plt.figure(figsize=(12,9))
    sns.heatmap(cm, annot=True)
    return fig
