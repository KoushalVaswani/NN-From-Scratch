import numpy as np


def confusion_matrix(y_true, y_pred, n_classes):
    """cm[i, j] = number of samples whose true class is i and predicted class is j."""
    cm = np.zeros((n_classes, n_classes), dtype=int)
    np.add.at(cm, (np.asarray(y_true, int), np.asarray(y_pred, int)), 1)
    return cm


def classification_report(cm):
    """Per-class precision, recall, F1 and support, plus the macro average of each."""
    tp = np.diag(cm).astype(float)
    predicted = cm.sum(axis=0)          # column sums: how often each class was predicted
    actual = cm.sum(axis=1)             # row sums: how often each class really occurs
    precision = np.divide(tp, predicted, out=np.zeros_like(tp), where=predicted > 0)
    recall = np.divide(tp, actual, out=np.zeros_like(tp), where=actual > 0)
    denom = precision + recall
    f1 = np.divide(2 * precision * recall, denom, out=np.zeros_like(tp), where=denom > 0)
    return dict(precision=precision, recall=recall, f1=f1, support=actual,
                macro=dict(precision=precision.mean(), recall=recall.mean(), f1=f1.mean()))
