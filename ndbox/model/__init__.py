from copy import deepcopy

from ndbox.utils import get_root_logger, MODEL_REGISTRY
from .wiener_filter import WienerFilterRegression, WienerFilterClassification
from .wiener_cascade import WienerCascadeRegression
from .kalman_filter import KalmanFilterRegression
from .support_vector_regression import SupportVectorRegression

__all__ = [
    'build_model',
    # wiener_filter.py
    'WienerFilterRegression',
    'WienerFilterClassification',
    # wiener_cascade.py
    'WienerCascadeRegression',
    # kalman_filter.py
    'KalmanFilterRegression',
    # support_vector_regression.py
    'SupportVectorRegression',
]


def build_model(opt):
    """
    Build model from options.

    :param opt: dict. Configuration. It must contain:
        type - str. Model type.
    :return: The specified model.
    """

    opt = deepcopy(opt)
    model = MODEL_REGISTRY.get(opt['type'])(**opt)
    logger = get_root_logger()
    logger.info(f'Model [{model.__class__.__name__}] is created.')
    return model
