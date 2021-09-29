import numpy as np
import sklearn
import config
import pandas as pd
from config import label_list
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import wandb

def get_confusion_matrix(labels, preds):
    """draw confusion matrix in wandb"""
    cm = confusion_matrix(labels, preds)
    norm_cm = cm / np.sum(cm, axis=1)[:,None]
    relation_class = label_list
    cm = pd.DataFrame(norm_cm, index=relation_class, columns=relation_class)
    fig = plt.figure(figsize=(12,9))
    sns.heatmap(cm, annot=True)
    
    return fig

def klue_re_micro_f1(preds, labels):
    """KLUE-RE micro f1 (except no_relation)"""
    no_relation_label_idx = label_list.index("no_relation")
    label_indices = list(range(len(label_list)))
    label_indices.remove(no_relation_label_idx)

    return sklearn.metrics.f1_score(labels, preds, average="micro", labels=label_indices) * 100.0

def klue_re_auprc(probs, labels):
    """KLUE-RE AUPRC (with no_relation)"""
    labels = np.eye(30)[labels]

    score = np.zeros((30,))
    for c in range(30):
        targets_c = labels.take([c], axis=1).ravel()
        preds_c = probs.take([c], axis=1).ravel()
        precision, recall, _ = sklearn.metrics.precision_recall_curve(targets_c, preds_c)
        score[c] = sklearn.metrics.auc(recall, precision)
    return np.average(score) * 100.0

def compute_metrics(pred):
    """ validation을 위한 metrics function """
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    probs = pred.predictions

  # calculate accuracy using sklearn's function
    f1 = klue_re_micro_f1(preds, labels)
    auprc = klue_re_auprc(probs, labels)
    acc = accuracy_score(labels, preds) # 리더보드 평가에는 포함되지 않습니다.

    cm_fig = get_confusion_matrix(labels, preds)
    wandb.log({'confusion matrix': wandb.Image(cm_fig)})
    
    return {
      'micro f1 score': f1,
      'auprc' : auprc,
      'accuracy': acc,
    }