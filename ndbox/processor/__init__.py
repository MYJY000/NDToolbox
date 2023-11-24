from copy import deepcopy

from ndbox.utils import PROCESSOR_REGISTRY
from .resample import resample
from .smooth import gaussian_smooth

__all__ = [
    # resample.py
    'resample',
    # smooth.py
    'gaussian_smooth',
    # split.py
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
    PROCESSOR_REGISTRY.get(processor_type)(nwb_data, **opt)
