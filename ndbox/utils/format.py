import numpy as np
from copy import deepcopy

from ndbox.utils import FORMAT_REGISTRY


def run_format(x, y, opt):
    opt = deepcopy(opt)
    format_type = opt.pop('type')
    format_x, format_y = FORMAT_REGISTRY.get(format_type)(x, y, **opt)
    return format_x, format_y


@FORMAT_REGISTRY.register()
def spike_rnn_format(x, y, bins_before, bins_after, bins_current=1, **kwargs):
    num_bins = bins_before + bins_after + bins_current
    num_examples = x.shape[0] - bins_before - bins_after
    num_neurons = x.shape[1]
    format_x = np.empty([num_examples, num_bins, num_neurons])
    for inx in range(num_examples):
        format_x[inx, :, :] = x[inx:(inx + num_bins), :]
    format_y = deepcopy(y)
    format_y = format_y[bins_before:(y.shape[0] - bins_after)]
    return format_x, format_y
