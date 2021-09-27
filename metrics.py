import numpy as np
from sklearn.metrics import f1_score, precision_recall_curve, auc


def make_compute_metrics(label_indices):
    n_classes = len(label_indices)
    no_relation_label_idx = label_indices.index("no_relation")
    label_indices = list(range(len(label_indices)))
    label_indices.remove(no_relation_label_idx)
    
    def compute_metrics(eval_pred, label_indices=label_indices, n_classes=n_classes):
        preds, labels = eval_pred

        # Micro F1 (except no_relation)
        predictions = np.argmax(preds, axis=1).ravel()
        micro_f1 = f1_score(labels, predictions, average="micro", labels=label_indices)

        # AUPRC (Area Under the Precision-Recall Curve)
        onehots = np.eye(n_classes)[labels]
        scores = np.zeros((n_classes,))
        for c in range(n_classes):
            targets_c = onehots.take([c], axis=1).ravel()
            preds_c = preds.take([c], axis=1).ravel()
            precision, recall, _ = precision_recall_curve(targets_c, preds_c)
            scores[c] = auc(recall, precision)
        auprc = np.average(scores)

        return {
            "micro_f1": micro_f1,
            "auprc": auprc,
        }
    
    return compute_metrics