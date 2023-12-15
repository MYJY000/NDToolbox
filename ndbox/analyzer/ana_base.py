from ndbox.utils.logger import get_root_logger
import numpy as np
from typing import List, Tuple

class Analyzer:
    name: str = None
    ana_type: str = None
    description: str = None

    def __init__(self, logger_name=None, params_data=None, params_plot=None):
        """
        A basic analyzer, has some basic self description.

        Parameters
        ------
        logger_name: str
            Name of the logger
        params_data: dict
            Params to process the data
        params_plot: dict
            Params to visualize the data
        """
        self.params_data = params_data
        self.params_plot = params_plot
        if logger_name is None:
            self.logger = get_root_logger()
        else:
            self.logger = get_root_logger(logger_name=logger_name)

    def process(self, *args, **kwargs):
        pass

    def plot(self, *args, **kwargs):
        pass

    def set_params_data(self, params_data: dict):
        self.params_data = params_data

    def set_params_plot(self, params_plot: dict):
        self.params_plot = params_plot

    def get_name(self):
        return self.name

    def get_ana_type(self):
        return self.ana_type

    def get_description(self):
        return self.description

    def get_params_data(self):
        return self.params_data

    def get_params_plot(self):
        return self.params_plot


class Neuron:
    PRECISION = 4
    precision = 10 ** -PRECISION

    def __init__(self, spike_train):
        """
        Basic Single Nueron data analyzing class.

        Notes
        ------
        All timestamps' unit is second(sec).

        Parameters
        ----------
        spike_train: np.ndarray
            1-D array, it means: The spike timestamps of a neuron, not binned.
        """
        # All attributes here
        self.spike_train = spike_train
        self.empty = spike_train.size == 0
        if spike_train is not None and not self.empty:
            self.t_start = np.floor(np.min(spike_train))
            self.t_stop = np.ceil(np.max(spike_train))
            self.duration = self.t_stop - self.t_start
        else:
            self.t_start = 0
            self.t_stop = 0
            self.duration = 0
        self.logger = get_root_logger()

    def cut(self, t_start=None, t_stop=None, inplace=False):
        """
        Cut the original firing timestamps, only retain firing timestamps in [t_start, t_stop).

        Parameters
        ------
        t_start: float
            The sample beginning timestamp.
        t_stop: float
            The sample finishing timestamp.
        inplace: bool
            If the cut operation update the original spike train.
        Note
        ------
        1. You may make sure that (t_stop-t_start)/bin_size is an integer.

        Returns
        ------
        np.ndarray
            spike_train: np.ndarray
                The neurons' firing timestamps in [t_start, t_stop).
        """
        if self.empty:
            return self.spike_train
        if t_start is None:
            t_start = self.t_start
        if t_stop is None:
            t_stop = self.t_stop
        self.flt = (self.spike_train >= t_start) & (self.spike_train < t_stop)
        spike_train = self.spike_train[self.flt]
        if inplace:
            self.t_start = round(t_start, self.PRECISION)
            if t_stop != np.inf:
                self.t_stop = round(t_stop, self.PRECISION)
            self.duration = round(self.t_stop - self.t_start, self.PRECISION)
            self.spike_train = spike_train
        return spike_train

    def sample(self, bin_size, smooth=None, kernel_size=None, **kwargs):
        """
        Sample the original firing timestamps with given `bin_size`, you can choose to smooth it.

        Parameters
        ------
        bin_size: float
            the sampling resolution, unit is s(sec)
        smooth: {'avg', 'gas'} or None
            use which kind of kernel to smooth the firing rates, default is None.
        kernel_size: int or None
            If `smooth` is not None, then `kernel_size` is valid. Generally,
            the bigger the `kernel_size`, the smoother the `firing_rates` is.
            Default 10 if `` is not None.

        Returns
        -----
            firing_rates: np.ndarray
                firing rates, firing_rates[i] is the firing rates of the i-th bin (i=0, 1, ..., n-1).
            bin_edges: np.ndarray
                time bin edges, the i-th bin is defined as `[bin_edges[i], bin_edges[i+1])` (i=0, 1, ..., n-1).
        """
        bc = self.duration / bin_size
        if bc - int(bc) > self.precision:
            self.logger.warning("The last bin's size is smaller, due to "
                                f"{self.duration} / {bin_size} is not an integer.")
        bin_edges = np.arange(self.t_start, self.t_stop, bin_size)
        if self.t_stop - bin_edges[-1] > self.precision:
            bin_edges = np.append(bin_edges, self.t_stop)
        hist, bin_edges = np.histogram(self.spike_train, bin_edges)
        firing_rates = hist / bin_size
        if smooth is not None:
            if kernel_size is None:
                kernel_size = 10
            if smooth == 'avg':
                firing_rates = np.convolve(firing_rates, self.__avg_kernel(kernel_size), 'same')
            elif smooth == 'gas':
                firing_rates = np.convolve(firing_rates, self.__gas_kernel(kernel_size), 'same')
            else:
                self.logger.warning(f"smooth={smooth} is not a valid value, the firing_rates "
                                    f"won't be smoothed. Note that `avg` or `gas` may be ok.")
        return np.round(firing_rates, self.PRECISION), np.round(bin_edges, self.PRECISION)

    def __avg_kernel(self, sz):
        return np.ones(int(sz)) / float(sz)

    def __gas_kernel(self, sz):
        n = 1000
        points = np.random.normal(size=n)
        points = points - points.min()
        points = points / points.max()
        window = [((points >= i / sz) & (points < (i + 1) / sz)).sum() / n for i in range(sz)]
        return window

    def split(self, stimulus, bias_start, bias_stop, bin_size=None, **kwargs):
        """
        Split the spike_train using given event(stimulus/mark/flag) time series, and return
        the split epochs and corresponding time bins. Epochs[i] is the spike_train timestamps
        around event_train[i], that is [event_train[i]-bias_start, event_train[i]+bias_stop).

        Parameters
        ------
        stimulus: np.ndarray
            The stimulus time sequence
        bias_start: float
            The time bias before the event.
        bias_stop: float
            The time bias after the event.
        bin_size: float
            the sampling resolution, unit is s(sec)

        Notes
        -------
        Other params like `smooth` `kernel_size` are allowed.

        Returns
        ------
        spike_trains: List[np.ndarray]
            The i-th of `spike_trains` is the firing timestamps in
            [event_train[i]-bias_start, event_train[i]+bias_stop)
        bin_edges: np.ndarray
            Time bin edges, the i-th bin is defined as `[bin_edges[i], bin_edges[i+1])`
            (i=0, 1, ..., n-1). When `bin_size` is not None
        epochs: np.ndarray
            For each sp in `spike_trains`, if `bin_size` is not None, sp
            will be sampled with `bin_size`
        """
        after_stop = stimulus + bias_stop >= self.t_stop
        if after_stop.any():
            self.logger.warning("Some event timestamps later than last firing stamp, they will be discard.")
        before_start = stimulus - bias_start < self.t_start
        if before_start.any():
            self.logger.warning("Some event timestamps earlier than first firing stamp, they will be discard.")
        stimulus = stimulus[(~before_start) & (~after_stop)]
        spike_trains = []
        epochs = []

        for i in range(len(stimulus)):
            sp = Neuron(self.spike_train)
            sp.cut(stimulus[i] - bias_start, stimulus[i] + bias_stop, inplace=True)
            spike_trains.append(sp.spike_train)
            if bin_size is not None:
                firing_rates, _ = sp.sample(bin_size, **kwargs)
                epochs.append(firing_rates)
        if bin_size is None:
            return spike_trains
        else:
            bin_edges = Neuron(self.spike_train)
            bin_edges.cut(-bias_start, bias_stop, True)
            bin_edges = bin_edges.sample(bin_size)[1]
            return spike_trains, bin_edges, np.array(epochs)


class NeuronList:
    def __init__(self, spike_trains):
        """
        Manage a list of neuron firing stamps

        Parameters
        -------
        spike_trains: List[np.ndarray]
            Each element represents a single neuron's firing timestamps
        """
        self.neurons = [Neuron(sp) for sp in spike_trains]
        self.spike_train = spike_trains
        self.spike_bins = None
        self.bin_size = None
        self.t_start = None
        self.t_stop = None

    def cut(self, t_start, t_stop, inplace=False):
        spike_trains = [sp.cut(t_start, t_stop, inplace=inplace) for sp in self.neurons]
        if inplace:
            self.t_start = t_start
            self.t_stop = t_stop
            self.spike_train = spike_trains
        return spike_trains

    def sample(self, bin_size, smooth=None, kernel_size=None):
        if self.spike_bins is not None and abs(bin_size-self.bin_size) < Neuron.precision:
            return self.spike_bins
        firing_rates = []
        bin_edges = []
        for sp in self.neurons:
            firing_rate, bin_edges = sp.sample(bin_size, smooth, kernel_size)
            firing_rates.append(firing_rate)
        self.spike_bins = np.array(firing_rates)
        return self.spike_bins, bin_edges

def plot_th(hist, bin_edges, axes, form='bar', **kwargs):
    """
    Plots a time histogram graph, and save the figure to given save path

    Parameters
    ------
    hist: np.ndarray
        The y value
    bin_edges: np.ndarray
        The bin edges
    form: str, {'bar', 'line', 'step', 'v-line', 'points}
        Default 'bar', choose from {'bar', 'curve', 'step', 'v-line', 'points'}
    axes: Axes or None
        Matplotlib axes handle. If None, new axes are created and returned.
    """
    if bin_edges.size <= 1:
        raise ValueError("Empty input.")
    bar_width = kwargs.pop('bar_width', bin_edges[1] - bin_edges[0])
    x = (bin_edges[:-1]+bin_edges[1:])/2
    if form == 'bar':
        axes.bar(x, hist, width=bar_width, **kwargs)
    elif form == 'line':
        axes.plot(x, hist, **kwargs)
    elif form == 'step':
        axes.step(x, hist, **kwargs)
    elif form == 'v-line':
        axes.vlines(x, np.zeros(hist.size), hist, **kwargs)
    elif form == 'points':
        axes.scatter(x, hist, **kwargs)
    else:
        raise ValueError("Output form invalid.")

