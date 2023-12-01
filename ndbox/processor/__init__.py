from copy import deepcopy

from ndbox.utils import PROCESSOR_REGISTRY
from .resample import resample, lag_offset
from .smooth import gaussian_smooth
from .split import train_test_split

__all__ = [
    'run_processor',
    # resample.py
    'resample',
    'lag_offset',
    # smooth.py
    'gaussian_smooth',
    # split.py
    'train_test_split',
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
