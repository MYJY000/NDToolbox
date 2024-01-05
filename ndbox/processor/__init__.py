from copy import deepcopy

from ndbox.utils import PROCESSOR_REGISTRY
from .resample import resample, lag_offset
from .smooth import gaussian_smooth
from .split import train_test_split, TRAIN_MASK, TEST_MASK, VAL_MASK

__all__ = [
    'run_processor',
    'build_processor',
    # resample.py
    'resample',
    'lag_offset',
    # smooth.py
    'gaussian_smooth',
    # split.py
    'train_test_split',
    'TRAIN_MASK', 'TEST_MASK', 'VAL_MASK',
]


def run_processor(nwb_data, opt):
    """
    Run processor for datasets.

    :param nwb_data: NWBDataset. Data.
    :param opt: dict. Configuration. It must contain:
        type - str. Processor type.
    """

    opt = deepcopy(opt)
    processor_type = opt.pop('type')
    PROCESSOR_REGISTRY.get(processor_type)(nwb_data=nwb_data, **opt)


def build_processor(nwb_data, processor_opt):
    for p_name, p_opt in processor_opt.items():
        if isinstance(p_opt, dict) and p_opt.get('type') is not None:
            run_processor(nwb_data, p_opt)
