import numpy as np
import pandas as pd
from copy import deepcopy

from ndbox.utils import PROCESSOR_REGISTRY


def float_equal(num1, num2, eps=0.001):
    if abs(num1 - num2) < eps:
        return True
    return False


@PROCESSOR_REGISTRY.register()
def resample(nwb_data, target_bin: float, **kwargs):
    nwb_data.logger.info(f"Resampling datasets to '{target_bin}' seconds.")
    if float_equal(target_bin, nwb_data.bin_size):
        return
    if not float_equal(target_bin % nwb_data.bin_size, 0):
        if nwb_data.spike_train is None:
            nwb_data.logger.error(f"There are no spike train in the dataset. "
                                  f"target_bin must be an integer multiple of "
                                  f"bin_size[{nwb_data.bin_size}].")
            return
        else:
            raise NotImplementedError('target_bin not be an integer multiple of '
                                      'bin_width is not supported in current version.')
    else:
        resample_factor = int(round(target_bin / nwb_data.bin_size))
        spike_mask = nwb_data.data.columns.str.startswith(nwb_data.spike_identifier)
        spike_columns = nwb_data.data.columns[spike_mask]
        other_columns = nwb_data.data.columns[~spike_mask]
        spike_arr = nwb_data.data[spike_columns].to_numpy()
        nan_mask = np.isnan(spike_arr[::resample_factor])
        remainder = spike_arr.shape[0] % resample_factor
        if remainder != 0:
            extra = spike_arr[-remainder:]
            spike_arr = spike_arr[:-remainder]
        else:
            extra = None
        spike_arr = np.nan_to_num(spike_arr, copy=False).reshape(
            (spike_arr.shape[0] // resample_factor, resample_factor, -1)
        ).sum(axis=1)
        if extra is not None:
            spike_arr = np.vstack([
                spike_arr, np.nan_to_num(extra, copy=False).sum(axis=0)
            ])
        spike_arr[nan_mask] = np.nan
        spike_df = pd.DataFrame(spike_arr, index=nwb_data.data.index[::resample_factor],
                                columns=spike_columns, dtype='float32')
        other_df = nwb_data.data[other_columns].iloc[::resample_factor]
        nwb_data.data = pd.concat([spike_df, other_df], axis=1)
        nwb_data.data.sort_index(axis=1, inplace=True)
    nwb_data.bin_size = target_bin


@PROCESSOR_REGISTRY.register()
def lag_offset(nwb_data, offset: float, **kwargs):
    nwb_data.logger.info(f"Setting offset {offset} seconds.")
    nwb_data.data = nwb_data.data.dropna()
    bin_size = nwb_data.bin_size
    lag_bins = int(round(offset / bin_size))
    spike_columns, other_columns = nwb_data.get_spike_and_other_columns()
    if lag_bins == 0:
        data_index = deepcopy(nwb_data.data.index[:])
    else:
        data_index = deepcopy(nwb_data.data.index[:(-2 * lag_bins)])
    nwb_data.data[spike_columns] = nwb_data.data[spike_columns].shift(-lag_bins)
    nwb_data.data[other_columns] = nwb_data.data[other_columns].shift(lag_bins)
    nwb_data.data = nwb_data.data.dropna()
    nwb_data.data.index = data_index
