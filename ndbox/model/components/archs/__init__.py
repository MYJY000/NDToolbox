from copy import deepcopy

from ndbox.utils import get_root_logger, ARCH_REGISTRY
from .mlp import MLP
from .LSTM import LSTM

__all__ = [
    'build_arch',
    # mlp.py
    'MLP',
    # LSTM.py
    'LSTM',
]


def build_arch(opt):
    opt = deepcopy(opt)
    arch = ARCH_REGISTRY.get(opt['type'])(**opt)
    logger = get_root_logger()
    logger.info(f'Arch [{arch.__class__.__name__}] is created.')
    return arch
