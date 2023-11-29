import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold

from ndbox.utils import PROCESSOR_REGISTRY


TRAIN_MASK = 0
TEST_MASK = 1
VAL_MASK = 2


@PROCESSOR_REGISTRY.register()
def train_test_split(nwb_data, train_size=None, test_size=None,
                     shuffle=False, stratify_target=None, **kwargs):
    indices = np.arange(len(nwb_data.data))
    if stratify_target is not None:
        if len(stratify_target) != 1:
            raise ValueError(f"stratify_target not support, should be None or list of length 1.")
        _, y = nwb_data.get_behavior_array(stratify_target)
        train_idx, test_idx, _, _ = train_test_split(
            indices, y, train_size=train_size, test_size=test_size,
            shuffle=shuffle, stratify=y
        )
    else:
        train_idx, test_idx = train_test_split(
            indices, train_size=train_size, test_size=test_size, shuffle=shuffle
        )

    mask = np.zeros(len(nwb_data.data), dtype=int)
    mask[train_idx] = TRAIN_MASK
    mask[test_idx] = TEST_MASK
    split_mask = pd.DataFrame(mask, index=nwb_data.index, columns=['split'])
    nwb_data.data = pd.concat([nwb_data.data, split_mask], axis=1)
    nwb_data.data.sort_index(axis=1, inplace=True)
