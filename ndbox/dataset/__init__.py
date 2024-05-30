from .nwb_dataset import NWBDataset
from .basic import NeuralDataset
from .hdf_dao import HierarchicalFileLoader
from .hdf_dataset import HDFNeuralDataset
from .dandi import DANDI
from .nohuman import NoHumanPR
from .spike_dataset import SpikeTrainDataset


__all__ = [
    'build_dataset',
    'NWBDataset',
    'NeuralDataset',
    'HierarchicalFileLoader',
    'HDFNeuralDataset',
    'DANDI',
    'NoHumanPR',
    'SpikeTrainDataset'
]


def build_dataset(opt):
    """
    Build dataset from options.

    :param opt: dict. Configuration. It must contain:
        name - str. Dataset name.
        type - str. Dataset type.
    """
    from ndbox.utils.logger import get_root_logger
    from ndbox.utils.registry import DATASET_REGISTRY
    from copy import deepcopy
    opt = deepcopy(opt)
    dataset = DATASET_REGISTRY.get(opt['type'])(**opt)
    logger = get_root_logger()
    logger.info(f"Dataset [{dataset.__class__.__name__}] - {opt['name']} is built.")
    return dataset
