import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import wandb
from functools import partial
from sklearn.metrics import precision_recall_curve, f1_score, auc, confusion_matrix
from .file_utils import RELATION_CLASS


def compute_micro_f1(logits, labels, label_indices=None):
    """ Compute Micro F1 score for specific labels """
    predictions = np.argmax(logits, axis=1).ravel()
    micro_f1 = f1_score(labels, predictions, 
                        average="micro", labels=label_indices)
    return micro_f1

    
def compute_auprc(probs, labels):
    """ Compute Area Under the Precision-Recall Curve """
    onehots = np.eye(N_CLASSES)[labels]
    scores = np.zeros((N_CLASSES,))
    for c in range(N_CLASSES):
        targets_c = onehots.take([c], axis=1).ravel()
        prods_c = probs.take([c], axis=1).ravel()
        precision, recall, _ = precision_recall_curve(targets_c, prods_c)
        scores[c] = auc(recall, precision)
    auprc = np.average(scores)
    return auprc


def get_confusion_matrix(logits, labels):
    """ Compute and Draw the Confusion Matrix """
    preds = np.argmax(logits, axis=1).ravel()
    cm = confusion_matrix(labels, preds)
    norm_cm = cm / np.sum(cm, axis=1)[:,None]
    cm = pd.DataFrame(norm_cm, index=RELATION_CLASS, columns=RELATION_CLASS)
    fig = plt.figure(figsize=(12,9))
    sns.heatmap(cm, annot=True)
    return fig


def compute_klue_re_leaderboard(eval_pred):
    """ Compute the KLUE-RE leaderboard metrics """
    # Parsing predictions and labels
    preds, labels = eval_pred
    # Preprocess no_relation for micro F1
    no_relation_label_idx = RELATION_CLASS.index("no_relation")
    label_indices = list(range(len(RELATION_CLASS)))
    label_indices.remove(no_relation_label_idx)
    # Compute micro F1
    micro_f1 = compute_micro_f1(preds, labels, label_indices)
    # Compute AURPC
    auprc = compute_auprc(preds, labels)
    # Compute confusion matrix
    cm_fig = get_confusion_matrix(preds, labels)
    try:
        wandb.log({'confusion matrix': wandb.Image(cm_fig)})
    except:
        print("Warning (원래는 Error): You must call wandb.init() before wandb.log()")
    return {
        "micro_f1": micro_f1,
        "auprc": auprc,
    }


N_CLASSES = len(RELATION_CLASS)

TASK_METRIC_MAP = {
    "klue_re": compute_klue_re_leaderboard,
    "tapt": None,
    "klue_re_type": compute_klue_re_leaderboard,
}