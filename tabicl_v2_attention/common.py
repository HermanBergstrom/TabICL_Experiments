import numpy as np


def to_numpy_array(value):
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        return value.numpy()
    return np.asarray(value)


def to_label_array(y):
    if hasattr(y, "reset_index"):
        y = y.reset_index(drop=True)
    if hasattr(y, "to_numpy"):
        return y.to_numpy()
    return np.asarray(y)


def adjust_probs_for_single_class(y_probs, y_labels, pos_label):
    unique_classes = np.unique(y_labels)
    if len(unique_classes) < 2:
        if unique_classes[0] == pos_label:
            return np.column_stack((1 - y_probs[:, 0], y_probs[:, 0]))
        return np.column_stack((y_probs[:, 0], 1 - y_probs[:, 0]))
    return y_probs
