from copy import deepcopy

import torch
from ndbox.utils import METRIC_REGISTRY
from .regression_metric import calculate_R2, calculate_cc, calculate_mae, calculate_mse, calculate_rmse
from .classification_metric import (calculate_precision, calculate_accuracy, calculate_recall,
                                    calculate_confusion_matrix)

__all__ = [
    'calculate_metric',
    'build_metric',
    # regression_metric.py
    'calculate_R2',
    'calculate_cc',
    'calculate_mae',
    'calculate_mse',
    'calculate_rmse',
    # classification_metric.py
    'calculate_accuracy',
    'calculate_recall',
    'calculate_precision',
    'calculate_confusion_matrix',
]


def calculate_metric(y_true, y_pred, opt):
    """
    Calculate metric from datasets and options.

    :param y_true: The true outputs (a matrix of size
        number of examples x number of outputs).
    :param y_pred: The predicted outputs (a matrix of
        size number of examples x number of outputs).
    :param opt: dict. Configuration. It must contain:
        type - str. Metric type.
    :return: The result of metric.
    """

    opt = deepcopy(opt)
    metric_type = opt.pop('type')
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()
    metric = METRIC_REGISTRY.get(metric_type)(y_true, y_pred, **opt)
    return metric


def build_metric(y_true, y_pred, metric_list):
    metric_dict = {}
    for metric_name, metric_opt in metric_list.items():
        metric_value = calculate_metric(y_true, y_pred, metric_opt)
        metric_dict[metric_name] = metric_value
    return metric_dict
