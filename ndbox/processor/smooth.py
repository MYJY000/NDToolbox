import numpy as np
import pandas as pd
from scipy import signal

from ndbox.utils import PROCESSOR_REGISTRY


@PROCESSOR_REGISTRY.register()
def gaussian_smooth(nwb_data, gauss_width: float, ignore_nans: bool = False, **kwargs):
    nwb_data.logger.info(f"Smoothing spikes with a {gauss_width} seconds Gaussian.")

    nwb_data.drop_smooth_columns()

    gauss_bin_std = gauss_width / nwb_data.bin_size
    win_len = int(6 * gauss_bin_std)
    window = signal.windows.gaussian(win_len, gauss_bin_std)
    window /= np.sum(window)
    spike_columns, spike_vals = nwb_data.get_spike_array()

    smoothed_spikes = np.apply_along_axis(
        lambda x: smooth_1d(x, window, ignore_nans), 0, spike_vals
    )

    smooth_columns = [nwb_data.smooth_identifier + c for c in spike_columns]
    smoothed_df = pd.DataFrame(smoothed_spikes, index=nwb_data.data.index,
                               columns=smooth_columns)
    nwb_data.data = pd.concat([nwb_data.data, smoothed_df], axis=1)
    nwb_data.data.sort_index(axis=1, inplace=True)


def rectify(arr):
    arr[arr < 0] = 0
    return arr


def smooth_1d(x, window, ignore_nans):
    if ignore_nans and np.any(np.isnan(x)):
        x.astype('float64')
        splits = np.where(np.diff(np.isnan(x)))[0] + 1
        seqs = np.split(x, splits)
        seqs = [seq if np.any(np.isnan(seq)) else
                rectify(signal.convolve(seq, window, 'same')) for seq in seqs]
        y = np.concatenate(seqs)
    else:
        y = signal.convolve(x.astype('float64'), window, 'same')
    return y
