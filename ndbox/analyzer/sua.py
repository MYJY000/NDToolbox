from .ana_base import Analyzer, Neuron, plot_th
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

class TimeHistAnalyzer(Analyzer):
    ana_type = 'plot_time_hist'
    description = 'Plot the time histogram of spike train.'
    name = "Time Histogram Analysis"

    def __init__(self, spike_train: np.ndarray, logger_name=None):
        super().__init__(logger_name=logger_name)
        self.data = Neuron(spike_train)
        self.result = None

    def process(self, t_start=None, t_stop=None, bin_size=None, smooth=None, kernel_size=None):
        if t_start is None:
            t_start = self.data.t_start
        if t_stop is None:
            t_stop = self.data.t_stop
        if bin_size is None:
            bin_size = 0.005
        self.data.cut(t_start, t_stop, inplace=True)
        firing_rates, bin_edges = self.data.sample(bin_size, smooth, kernel_size)
        self.set_params_data({
            't_start': t_start,
            't_stop': t_stop,
            'bin_size': bin_size,
            'smooth': smooth,
            'kernel_size': kernel_size
        })
        self.result = {
            'firing_rates': firing_rates,
            'bin_edges': bin_edges
        }
        return self.result

    def plot(self, axes=None, form='bar', **kwargs):
        if self.result is None:
            self.logger.warning("Please process the data first!")
            return
        if axes is None:
            fig, axes = plt.subplots()
        plot_th(self.result['firing_rates'], self.result['bin_edges'], axes, form, **kwargs)
        kwargs['form'] = form
        self.set_params_plot(kwargs)
        return axes

class CumActivityAnalyzer(Analyzer):
    ana_type = 'plot_cum_activity'
    description = 'Plot the cumulative activity of spike train. When a spike found, the curve step up.'
    name = "Cumulative Activity Analysis"

    def __init__(self, spike_train: np.ndarray, logger_name=None):
        super().__init__(logger_name=logger_name)
        self.data = Neuron(spike_train)
        self.result = None

    def process(self, t_start=None, t_stop=None, bin_size=None):
        if t_start is None:
            t_start = self.data.t_start
        if t_stop is None:
            t_stop = self.data.t_stop
        if bin_size is None:
            bin_size = 0.005
        self.data.cut(t_start, t_stop, inplace=True)
        firing_rates, bin_edges = self.data.sample(bin_size)
        self.set_params_data({
            't_start': t_start,
            't_stop': t_stop,
            'bin_size': bin_size,
        })
        self.result = {
            'cum_activity': np.cumsum(firing_rates) * bin_size,
            'bin_edges': bin_edges
        }
        return self.result

    def plot(self, axes=None, **kwargs):
        if self.result is None:
            self.logger.warning("Please process the data first!")
            return
        if axes is None:
            fig, axes = plt.subplots()
        plot_th(self.result['cum_activity'], self.result['bin_edges'], axes, 'step', **kwargs)
        self.set_params_plot(kwargs)
        return axes

class RasterAnalyzer(Analyzer):
    ana_type = 'plot_raster'
    description = 'Raster plot of a single neuron.'
    name = "Raster Plot Analysis"

    def __init__(self, spike_train: np.ndarray, logger_name=None):
        super().__init__(logger_name=logger_name)
        self.data = Neuron(spike_train)
        self.result = None

    def process(self, t_start=None, t_stop=None):
        if t_start is None:
            t_start = self.data.t_start
        if t_stop is None:
            t_stop = self.data.t_stop
        self.data.cut(t_start, t_stop, inplace=True)
        self.set_params_data({
            't_start': t_start,
            't_stop': t_stop,
        })
        self.result = {
            'raster': self.data.spike_train
        }
        return self.result

    def plot(self, axes=None, **kwargs):
        if self.result is None:
            self.logger.warning("Please process the data first!")
            return
        if axes is None:
            fig, axes = plt.subplots()
        linelengths = kwargs.pop("linelengths", 0.86)
        linewidths = kwargs.pop("linewidths", 0.8)
        axes.eventplot(self.data.spike_train, linelengths=linelengths,
                       linewidths=linewidths, **kwargs)
        kwargs["linelengths"] = linelengths
        kwargs["linewidths"] = linewidths
        self.set_params_plot(kwargs)
        return axes

class ISIAnalyzer(Analyzer):
    ana_type = 'plot_isi_distribution'
    description = 'Plot the inter spike interval distribution of spike train.'
    name = "Inter Spike Interval Distribution Analysis"

    def __init__(self, spike_train: np.ndarray, logger_name=None):
        super().__init__(logger_name=logger_name)
        self.data = Neuron(spike_train)
        self.result = None

    def process(self, bin_size=None, t_start=None, t_stop=None, min_width=None, max_width=None):
        if t_start is None:
            t_start = self.data.t_start
        if t_stop is None:
            t_stop = self.data.t_stop
        if bin_size is None:
            bin_size = 0.005
        if min_width is None:
            min_width = 0
        if max_width is None:
            max_width = 0.5
        self.data.cut(t_start, t_stop, inplace=True)
        interval = Neuron(np.diff(self.data.spike_train))
        if interval.empty:
            self.logger.warning("No intervals found!")
            return
        interval.cut(min_width, max_width, True)
        isi_dis, isi_edges = interval.sample(bin_size)

        self.set_params_data({
            't_start': t_start,
            't_stop': t_stop,
            'bin_size': bin_size,
            'min_width': 0,
            'max_width': 0.5
        })
        self.result = {
            'interval': interval.spike_train,
            'isi_dis': isi_dis,
            'isi_edges': isi_edges
        }
        return self.result

    def plot(self, axes=None, form='bar', **kwargs):
        if self.result is None:
            self.logger.warning("Please process the data first!")
            return
        if axes is None:
            fig, axes = plt.subplots()
        plot_th(self.result['isi_dis'], self.result['isi_edges'], axes, form, **kwargs)
        kwargs['form'] = form
        self.set_params_plot(kwargs)
        return axes

class ISITimeHistAnalyzer(Analyzer):
    ana_type = 'plot_isi_time_hist'
    description = 'Plot the inter spike interval time histogram of spike train.'
    name = "Inter Spike Interval Time Analysis"

    def __init__(self, spike_train: np.ndarray, logger_name=None):
        super().__init__(logger_name=logger_name)
        self.data = Neuron(spike_train)
        self.result = None

    def process(self, t_start=None, t_stop=None, min_width=None, max_width=None):
        if t_start is None:
            t_start = self.data.t_start
        if t_stop is None:
            t_stop = self.data.t_stop
        if min_width is None:
            min_width = 0
        if max_width is None:
            max_width = 0.5
        self.set_params_data({
            't_start': t_start,
            't_stop': t_stop,
            'min_width': 0,
            'max_width': 0.5
        })

        self.data.cut(t_start, t_stop, inplace=True)
        interval = Neuron(np.diff(self.data.spike_train))
        if interval.empty:
            self.logger.warning("No intervals found!")
            return
        interval.cut(min_width, max_width, True)
        bin_edges = self.data.spike_train[:-1][interval.flt]
        bin_edges = np.append(bin_edges, self.data.spike_train[-1])
        self.result = {
            'interval': interval.spike_train,
            'isi_time_hist': interval.spike_train,
            'bin_edges': bin_edges
        }
        return self.result

    def plot(self, axes=None, form='bar', **kwargs):
        if self.result is None:
            self.logger.warning("Please process the data first!")
            return
        if axes is None:
            fig, axes = plt.subplots()
        plot_th(self.result['isi_time_hist'], self.result['bin_edges'], axes, form, **kwargs)
        kwargs['form'] = form
        self.set_params_plot(kwargs)
        return axes

class InstantFreqAnalyzer(Analyzer):
    ana_type = 'plot_instant_freq'
    description = 'Plot the instantaneous frequency of spike train.'
    name = "Instantaneous Frequency Time Analysis"

    def __init__(self, spike_train: np.ndarray, logger_name=None):
        super().__init__(logger_name=logger_name)
        self.data = Neuron(spike_train)
        self.result = None

    def process(self, t_start=None, t_stop=None, min_width=None, max_width=None):
        if t_start is None:
            t_start = self.data.t_start
        if t_stop is None:
            t_stop = self.data.t_stop
        if min_width is None:
            min_width = 0
        if max_width is None:
            max_width = 0.5
        self.set_params_data({
            't_start': t_start,
            't_stop': t_stop,
            'min_width': 0,
            'max_width': 0.5
        })

        self.data.cut(t_start, t_stop, inplace=True)
        interval = Neuron(np.diff(self.data.spike_train))
        if interval.empty:
            self.logger.warning("No intervals found!")
            return
        interval.cut(min_width, max_width, True)
        bin_edges = self.data.spike_train[:-1][interval.flt]
        bin_edges = np.append(bin_edges, self.data.spike_train[-1])
        self.result = {
            'interval': interval.spike_train,
            'instant_freq': 1/interval.spike_train,
            'bin_edges': bin_edges
        }
        return self.result

    def plot(self, axes=None, **kwargs):
        if self.result is None:
            self.logger.warning("Please process the data first!")
            return
        if axes is None:
            fig, axes = plt.subplots()
        plot_th(self.result['instant_freq'], self.result['bin_edges'], axes, 'v-line', **kwargs)
        self.set_params_plot(kwargs)
        return axes

class PoincareMapAnalyzer(Analyzer):
    ana_type = 'plot_poincare_map'
    description = 'Poincare map of spike train, which is a scatter plot represents ' \
                  'the relation between adjacent intervals'
    name = "Poincare Map Analysis"

    def __init__(self, spike_train: np.ndarray, logger_name=None):
        super().__init__(logger_name=logger_name)
        self.data = Neuron(spike_train)
        self.result = None

    def process(self, t_start=None, t_stop=None, min_width=None, max_width=None):
        if t_start is None:
            t_start = self.data.t_start
        if t_stop is None:
            t_stop = self.data.t_stop
        if min_width is None:
            min_width = 0
        if max_width is None:
            max_width = 0.5
        self.set_params_data({
            't_start': t_start,
            't_stop': t_stop,
            'min_width': 0,
            'max_width': 0.5
        })

        self.data.cut(t_start, t_stop, inplace=True)
        interval = Neuron(np.diff(self.data.spike_train))
        if interval.empty:
            self.logger.warning("No intervals found!")
            return
        interval.cut(min_width, max_width, True)
        poincare_x = interval.spike_train[1:]
        poincare_y = interval.spike_train[:-1]
        self.result = {
            'poincare_x': poincare_x,
            'poincare_y': poincare_y,
        }
        return self.result

    def plot(self, axes=None, **kwargs):
        if self.result is None:
            self.logger.warning("Please process the data first!")
            return
        if axes is None:
            fig, axes = plt.subplots()
        axes.scatter(self.result['poincare_x'], self.result['poincare_y'], **kwargs)
        self.set_params_plot(kwargs)
        return axes

class AutocorrelogramAnalyzer(Analyzer):
    ana_type = 'plot_autocorrelogram'
    description = 'Autocorrelogram shows the conditional probability of a spike at time t0+t ' \
                  'on the condition that there is a spike at time t0.'
    name = 'Auto-correlation Analysis'

    def __init__(self, spike_train: np.ndarray, logger_name=None):
        super().__init__(logger_name=logger_name)
        self.data = Neuron(spike_train)
        self.result = None

    def process(self, bias_start=None, bias_stop=None, bin_size=None,
                t_start=None, t_stop=None, smooth=None, kernel_size=None):
        if t_start is None:
            t_start = self.data.t_start
        if t_stop is None:
            t_stop = self.data.t_stop
        if bias_start is None:
            bias_start = 0.2
        if bias_stop is None:
            bias_stop = 0.2
        if bin_size is None:
            bin_size = 0.005
        self.set_params_data({
            't_start': t_start,
            't_stop': t_stop,
            'bias_start': bias_start,
            'bias_stop': bias_stop,
            'bin_size': bin_size,
            'smooth': smooth,
            'kernel_size': kernel_size
        })

        self.data.cut(t_start, t_stop, inplace=True)
        epochs, bin_edges, bin_epochs = self.data.split(self.data.spike_train, bias_start, bias_stop,
                                                        bin_size, smooth=smooth, kernel_size=kernel_size)
        self.result = {
            'auto_correlation_hist': np.average(bin_epochs, axis=0),
            'bin_edges': bin_edges
        }
        return self.result

    def plot(self, axes=None, form='bar', **kwargs):
        if self.result is None:
            self.logger.warning("Please process the data first!")
            return
        if axes is None:
            fig, axes = plt.subplots()
        plot_th(self.result['auto_correlation_hist'], self.result['bin_edges'], axes, form, **kwargs)
        kwargs['form'] = form
        self.set_params_plot(kwargs)
        return axes

