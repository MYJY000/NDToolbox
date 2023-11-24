import numpy as np
import pandas as pd
from scipy import signal

from ndbox.utils import PROCESSOR_REGISTRY


@PROCESSOR_REGISTRY.register()
def split_bins_train_val(nwb_data):
    raise NotImplementedError


def dataloader():
    pass
