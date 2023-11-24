import numpy as np
import pandas as pd

from ndbox.utils import PROCESSOR_REGISTRY


@PROCESSOR_REGISTRY.register()
def resample(nwb_data, target_bin, **kwargs):
    nwb_data.logger.info(f"Resampling datasets to '{target_bin}' seconds.")
    if target_bin == nwb_data.bin_size:
        return
    if target_bin % nwb_data.bin_size != 0:
        if nwb_data.spike_train is None:
            nwb_data.logger.error(f"There are no spike train in the dataset. "
                                  f"target_bin must be an integer multiple of "
                                  f"bin_size[{nwb_data.bin_size}].")
            return
        else:
            raise NotImplementedError
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
                                columns=spike_columns, dtype='float64')
        other_df = nwb_data.data[other_columns].iloc[::resample_factor]
        nwb_data.data = pd.concat([spike_df, other_df], axis=1)
        nwb_data.data.sort_index(axis=1, inplace=True)
