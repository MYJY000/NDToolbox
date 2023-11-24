from copy import deepcopy

from ndbox.utils import get_root_logger, DATASET_REGISTRY
from .nwb_dataset import NWBDataset

__all__ = [
    'build_dataset',
    # nwb_dataset.py
    'NWBDataset'
]


def build_dataset(opt):
    """
    Build dataset from options.

    :param opt: dict. Configuration. It must contain:
        name - str. Dataset name.
        type - str. Dataset type.
    """

    opt = deepcopy(opt)
    dataset = DATASET_REGISTRY.get(opt['type'])(**opt)
    logger = get_root_logger()
    logger.info(f"Dataset [{dataset.__class__.__name__}] - {opt['name']} is built.")
    return dataset
