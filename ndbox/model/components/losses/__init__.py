from copy import deepcopy

from ndbox.utils import get_root_logger, LOSS_REGISTRY
from .basic_loss import L1Loss, MSELoss

__all__ = [
    'build_loss',
    # basic_loss.py
    'L1Loss',
    'MSELoss'
]


def build_loss(opt):
    opt = deepcopy(opt)
    loss = LOSS_REGISTRY.get(opt['type'])(**opt)
    logger = get_root_logger()
    logger.info(f'Loss [{loss.__class__.__name__}] is create.')
    return loss
