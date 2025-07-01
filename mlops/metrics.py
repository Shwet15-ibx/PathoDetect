from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score, accuracy_score
import numpy as np

def calculate_metrics(y_true, y_pred, y_prob=None):
    metrics = {}
    metrics['f1_score'] = f1_score(y_true, y_pred)
    if y_prob is not None:
        metrics['auc'] = roc_auc_score(y_true, y_prob)
    metrics['precision'] = precision_score(y_true, y_pred)
    metrics['recall'] = recall_score(y_true, y_pred)
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    return metrics 