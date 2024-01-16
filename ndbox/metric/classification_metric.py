from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

from ndbox.utils import METRIC_REGISTRY


@METRIC_REGISTRY.register()
def calculate_accuracy(y_true, y_pred, normalize=True, sample_weight=None, **kwargs):
    return accuracy_score(y_true, y_pred, normalize=normalize, sample_weight=sample_weight)


@METRIC_REGISTRY.register()
def calculate_precision(y_true, y_pred, labels=None, pos_label=1, average='binary',
                        sample_weight=None, zero_division='warn', **kwargs):
    return precision_score(y_true, y_pred, labels=labels, pos_label=pos_label, average=average,
                           sample_weight=sample_weight, zero_division=zero_division)


@METRIC_REGISTRY.register()
def calculate_recall(y_true, y_pred, labels=None, pos_label=1, average='binary',
                     sample_weight=None, zero_division='warn', **kwargs):
    return recall_score(y_true, y_pred, labels=labels, pos_label=pos_label, average=average,
                        sample_weight=sample_weight, zero_division=zero_division)


@METRIC_REGISTRY.register()
def calculate_confusion_matrix(y_true, y_pred, labels=None, sample_weight=None,
                               normalize=True, **kwargs):
    return confusion_matrix(y_true, y_pred, labels=labels, sample_weight=sample_weight,
                            normalize=normalize)
