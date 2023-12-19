import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold

from ndbox.utils import PROCESSOR_REGISTRY

TRAIN_MASK = 1
TEST_MASK = 2
VAL_MASK = 3


@PROCESSOR_REGISTRY.register()
def train_test_bins_split(nwb_data, train_size: float = None, test_size: float = None,
                          shuffle: bool = False, stratify_target: list = None, idx: str = '', **kwargs):
    nwb_data.logger.info(f"Train test bins split.")
    indices = np.arange(len(nwb_data.data))
    if stratify_target is not None:
        if len(stratify_target) != 1:
            raise ValueError(f"stratify_target not support, should be None or list of length 1.")
        _, y = nwb_data.get_behavior_array(stratify_target)
        y = y.flatten()
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
    split_mask = pd.DataFrame(mask, index=nwb_data.data.index,
                              columns=[nwb_data.split_identifier + str(idx)])
    nwb_data.data = pd.concat([nwb_data.data, split_mask], axis=1)
    nwb_data.data.sort_index(axis=1, inplace=True)


@PROCESSOR_REGISTRY.register()
def KFord_split(nwb_data, n_splits: int = 5, shuffle: bool = False, idx: str = 'kf', **kwargs):
    nwb_data.logger.info(f"KFord split.")
    kf = KFold(n_splits=n_splits, shuffle=shuffle)
    indices = np.arange(len(nwb_data.data))
    cnt = 0
    for train_idx, test_idx in kf.split(indices):
        mask = np.zeros(len(nwb_data.data), dtype=int)
        mask[train_idx] = TRAIN_MASK
        mask[test_idx] = TEST_MASK
        split_mask = pd.DataFrame(mask, index=nwb_data.data.index,
                                  columns=[nwb_data.split_identifier + f"{idx}_{cnt}"])
        nwb_data.data = pd.concat([nwb_data.data, split_mask], axis=1)
        cnt += 1
    nwb_data.data.sort_index(axis=1, inplace=True)
