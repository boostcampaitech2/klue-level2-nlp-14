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


def compute_klue_re_leaderboard(eval_pred):
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
    return {
        "micro_f1": micro_f1,
        "auprc": auprc,
    }


N_CLASSES = len(RELATION_CLASS)

TASK_METRIC_MAP = {
    "klue_re": compute_klue_re_leaderboard,
    "tapt": None,
    "klue_re_type": compute_klue_re_leaderboard,
    "klue_re_entity_embedding": compute_klue_re_leaderboard,
}
