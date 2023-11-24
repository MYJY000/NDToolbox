import numpy as np

from ndbox.utils import METRIC_REGISTRY


@METRIC_REGISTRY.register()
def calculate_acc(y_true, y_pred, **kwargs):
    """
    Function to calculate accuracy.

    :param y_true: The true outputs (a matrix of size
        number of examples x number of outputs).
    :param y_pred: The predicted outputs (a matrix of
        size number of examples x number of outputs).
    :return: An array of accuracy for each output.
    """

    acc_array = np.mean(y_true == y_pred, axis=0)
    return acc_array
