import numpy as np


def classification_metrics(labels, predictions):
    matrix = np.zeros((4,4), dtype=np.int64)
    np.add.at(matrix, (np.asarray(labels,dtype=int), np.asarray(predictions,dtype=int)), 1)
    support = matrix.sum(1)
    diagonal = matrix.diagonal()
    denominator = matrix.sum(0)+support
    f1 = np.divide(2*diagonal, denominator, out=np.zeros(4,dtype=float), where=denominator>0)
    normalized = np.divide(matrix, support[:,None], out=np.zeros((4,4),dtype=float), where=support[:,None]>0)
    return dict(accuracy=float(diagonal.sum()/max(1,matrix.sum())), macro_f1=float(f1.mean()),
                per_class_f1=f1.tolist(), support=support.tolist(),
                confusion_matrix=matrix.tolist(), normalized_confusion_matrix=normalized.tolist())
