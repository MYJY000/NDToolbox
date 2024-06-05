"""
Correlative measures on spike trains
    `covariance`: Calculate the NxN matrix of pairwise covariances between all combinations of 
        N binned spike trains.
    `correlation_coefficient`: Calculate the NxN matrix of pairwise Pearson's correlation coefficients 
        between all combinations of N binned spike trains.
    `cross_correlation_histogram`: Computes the cross-correlation histogram (CCH) between two binned 
        spike trains binned_spiketrain_i and binned_spiketrain_j.
    `spike_time_tiling_coefficient`: Calculates the Spike Time Tiling Coefficient (STTC) as described 
        in (Cutts and Eglen, 2014) following their implementation in C.

Spike train dissimilarity
    `victor_purpura_distance`:
        Calculates the Victor-Purpura's (VP) distance.
    `van_rossum_distance`:
        Calculates the van Rossum distance (Rossum, 2001), defined as Euclidean distance of the spike trains 
        convolved with a causal decaying exponential smoothing filter.
    `spike_contrast`: Calculates the synchrony of spike trains, according to (Ciba et al., 2018).
    `Synchrotool`: Tool class to find, remove and/or annotate the presence of synchronous spiking 
        events across multiple spike trains.

autocorrelogram
jointpsth

"""


#
# import numpy as np
# from ndbox.utils.registry import ANALYZER_REGISTRY
# from typing import List, Tuple
# from matplotlib.pyplot import Axes
#
# @ANALYZER_REGISTRY.register()
# # https://elephant.readthedocs.io/en/latest/reference/_toctree/spike_train_synchrony/elephant.spike_train_synchrony.spike_contrast.html
# def scs_plot(spike_list, save_path, t_start=None, t_stop=None, bin_size=None, shrink=0.9,
#              axes=None, color='#646464', xlabel='Time bins(sec)', ylabel='Spike counts', title='Cross-correlograms',
#              **kwargs
#              ):
#     """
#     Spike-Contrast Synchrony plot. It calculate the synchrony of spike trains,
#     the spike trains can have different length. We use spike-contrast as a
#     measurement of spike-train synchrony.
#
#     Parameters
#     ----------
#     spike_list: List[np.ndarray] or np.ndarray
#         if data type is `List[np.ndarray]`, then spike_list[i] indicates the i-th neuron's
#         spike stamps, spike_list[i] is a 1D array.
#         if data type is np.ndarray, then it must be 2D array, spike_list[i] is the binned
#         firing counts of the i-th neuron. Note that you must pass the `bin_size` argument
#         in this way.
#     t_start: float
#         The record beginning timestamp.
#     t_stop: float
#         The record finishing timestamp.
#     bin_size: float
#         the bin width
#     shrink: float
#         A multiplier to shrink the bin size on each iteration.
#         The value must be in range (0, 1). Default: 0.9
#     save_path: str
#         The directory to store the figure.
#     axes: Axes or None
#         Matplotlib axes handle. If None, new axes are created and returned.
#     color: str or List[str]
#         Color of raster line, can be an array
#     xlabel: str
#         The label of x-axis
#     ylabel: str
#         The label of y-axis
#     title: str
#         The title of the plot
#
#     Returns
#     ------
#     A history of spike-contrast synchrony, computed for a range of different bin sizes,
#     alongside with the maximum value of the synchrony.
#
#     Tuple()
#         synchrony: float
#             Returns the synchrony of the input spike trains
#         trace_contrast: np.ndarray
#              the average sum of differences of the number of spikes in subsequent bins
#         trace_active: np.ndarray
#             the average number of spikes per bin, weighted by the number of spike trains
#             containing at least one spike inside the bin;
#         trace_synchrony: np.ndarray
#             the product of `trace_contrast` and `trace_active`;
#         bin_size: np.ndarray
#             the X axis, a list of bin sizes that correspond to these traces.
#
#
#     Raises
#     ------
#     ValueError
#         If bin_shrink_factor is not in (0, 1) range.
#         If the input spike trains contain no more than one spike-train.
#     TypeError
#         If the input spike trains is not a list or ndarray.
#     """
#     check_input(spike_list, bin_size)
#     # TODO
#     pass
#
#
# @ANALYZER_REGISTRY.register()
# # https://elephant.readthedocs.io/en/latest/reference/_toctree/spike_train_correlation/elephant.spike_train_correlation.correlation_coefficient.html
# def coefficient_plot(spike_list, save_path, t_start=None, t_stop=None, bin_size=None, shrink=0.9,
#                      axes=None, color='#646464', xlabel='Time bins(sec)', ylabel='Spike counts',
#                      title='Cross-correlograms',
#                      **kwargs
#                      ):
#     """
#     Correlation Coefficient plot. For each pair of spike trains, the correlation coefficient
#     is obtained by binning and at the desired bin size. For an input of N spike trains, an
#     N x N matrix is returned. Each entry in the matrix is a real number ranging between
#     -1 (perfectly anti-correlated spike trains) and +1 (perfectly correlated spike trains).
#
#     Parameters
#     ----------
#     spike_list: List[np.ndarray] or np.ndarray
#         if data type is `List[np.ndarray]`, then spike_list[i] indicates the i-th neuron's
#         spike stamps, spike_list[i] is a 1D array.
#         if data type is np.ndarray, then it must be 2D array, spike_list[i] is the binned
#         firing counts of the i-th neuron. Note that you must pass the `bin_size` argument
#         in this way.
#     t_start: float
#         The record beginning timestamp.
#     t_stop: float
#         The record finishing timestamp.
#     bin_size: float
#         the bin width
#     shrink: float
#         A multiplier to shrink the bin size on each iteration.
#         The value must be in range (0, 1). Default: 0.9
#     save_path: str
#         The directory to store the figure.
#     axes: Axes or None
#         Matplotlib axes handle. If None, new axes are created and returned.
#     color: str or List[str]
#         Color of raster line, can be an array
#     xlabel: str
#         The label of x-axis
#     ylabel: str
#         The label of y-axis
#     title: str
#         The title of the plot
#
#     Returns
#     ------
#     A history of spike-contrast synchrony, computed for a range of different bin sizes,
#     alongside with the maximum value of the synchrony.
#
#     dict
#         'synchrony': float
#             Returns the synchrony of the input spike trains
#         'trace_contrast': np.ndarray
#              the average sum of differences of the number of spikes in subsequent bins
#         'trace_active': np.ndarray
#             the average number of spikes per bin, weighted by the number of spike trains
#             containing at least one spike inside the bin;
#         'trace_synchrony': np.ndarray
#             the product of `trace_contrast` and `trace_active`;
#         'bin_size': np.ndarray
#             the X axis, a list of bin sizes that correspond to these traces.
#
#
#     Raises
#     ------
#     ValueError
#         If bin_shrink_factor is not in (0, 1) range.
#         If the input spike trains contain no more than one spike-train.
#     TypeError
#         If the input spike trains is not a list or ndarray.
#     """
#     check_input(spike_list, bin_size)
#     # TODO
#     pass
#
#
# @ANALYZER_REGISTRY.register()
# def spade_plot(spike_list, save_path, t_start=None, t_stop=None, bin_size=None, shrink=0.9,
#                axes=None, color='#646464', xlabel='Time bins(sec)', ylabel='Spike counts', title='Cross-correlograms',
#                **kwargs
#                ):
#     # TODO
#     pass
#
#
# # https://elephant.readthedocs.io/en/latest/reference/cell_assembly_detection.html
# def cad_plot():
#     # TODO
#     pass
#
#
# def check_input(spike_list, bin_size):
#     if isinstance(spike_list, np.ndarray):
#         if spike_list.ndim < 2:
#             raise TypeError("Get a numpy array. However, it's not 2-D.")
#         elif bin_size is None:
#             raise ValueError("Binned 2D array as input, however, bin_size is None.")
#         else:
#             raise NotImplementedError("to be continued ... ...")
#     if isinstance(spike_list, list):
#         if len(spike_list) < 2:
#             raise ValueError("Data list should have more than 1 spike trains.")
#         if not isinstance(spike_list[0], np.ndarray):
#             raise TypeError("A numpy array is expected.")
"""
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from .basic import Analyzer, Neuron, plot_th


class CrossCorrelationAnalyzer(Analyzer):
    "
    Relative analysis of `target` neuron relative to `reference` neuron  
    with consideration of `stimulus`(or we say `event`/`mark`).\nIt shows the
    conditional probability of a target spike at time `t0+t` on the condition that 
    there is a reference spike at time `t0` across trials.
    "
    def __init__(self, target: np.ndarray, refer: np.ndarray, event_train: np.ndarray, logger_name=None):
        super().__init__(logger_name=logger_name)
        self.target = Neuron(target)
        self.refer = Neuron(refer)
        self.event_train = event_train
        self.result = None

    def _shift_predictor(self, shift_predictor=None):
        trial_n, bin_n = self.target_bin_epochs.shape
        if shift_predictor is None:
            shift_predictor = [0]
        elif shift_predictor == 'random':
            shift_predictor = [0, np.random.randint(1, trial_n)]
        elif shift_predictor == 'average':
            shift_predictor = list(range(trial_n))
        else:
            raise ValueError(f"shift_predictor={shift_predictor} is not allowed")
        return shift_predictor

    def co_matrix(self, refer_bin_epochs):
        trial_n, bin_n = self.target_bin_epochs.shape
        trial_n2, bin_n2 = refer_bin_epochs.shape
        trial_n = min([trial_n, trial_n2])
        bin_n = min([bin_n, bin_n2])
        co_matrix = np.zeros((bin_n, bin_n))
        for i in range(trial_n):
            ta = self.target_bin_epochs[i]
            re = refer_bin_epochs[i]
            a = np.array([re for _ in re])
            for j in range(bin_n):
                a[j][a[j] > ta[j]] = ta[j]
            co_matrix += a
        return np.flipud(co_matrix)

    def process(self, shift_predictor=None, bias_start=None, bias_stop=None, bin_size=None,
                t_start=None, t_stop=None, smooth=None, kernel_size=None):
        if t_start is None:
            t_start = self.target.t_start
        if t_stop is None:
            t_stop = self.target.t_stop
        if bias_start is None:
            bias_start = 0.2
        if bias_stop is None:
            bias_stop = 0.2
        if bin_size is None:
            bin_size = 0.005

        self.set_params_data({
            'shift_predictor': shift_predictor,
            't_start': t_start,
            't_stop': t_stop,
            'bias_start': bias_start,
            'bias_stop': bias_stop,
            'bin_size': bin_size,
            'smooth': smooth,
            'kernel_size': kernel_size
        })
        self.target.cut(t_start, t_stop, True)
        t = self.target.split(self.event_train, bias_start, bias_stop, bin_size,
                              smooth=smooth, kernel_size=kernel_size)
        r = self.refer.split(self.event_train, bias_start, bias_stop, bin_size,
                             smooth=smooth, kernel_size=kernel_size)
        self.target_epochs, self.bin_edges, self.target_bin_epochs = t
        self.refer_epochs, self.bin_edges, self.refer_bin_epochs = r
        shift_predictor = self._shift_predictor(shift_predictor)
        ccg_list = []
        bin_edges = []
        trial_n, bin_n = self.refer_bin_epochs.shape
        for shp in shift_predictor:
            e_shp = np.roll(self.event_train, -shp) - self.event_train
            ref_train = []
            for i in range(trial_n):
                if self.refer_epochs[i].size > 0:
                    ref_train.append(self.refer_epochs[i] + e_shp[i])
            ref_train = np.concatenate(ref_train)
            refer_epochs, bin_edges, refer_bin_epochs = \
                self.target.split(ref_train, bias_start, bias_stop, bin_size,
                                  smooth=smooth, kernel_size=kernel_size)
            ccg_list.append(np.average(refer_bin_epochs, axis=0))
        shift_p = np.zeros(bin_n)
        if len(shift_predictor) > 1:
            shift_p = np.mean(np.array(ccg_list), axis=0)
        ccg_cor = ccg_list[0] - shift_p
        self.result = {
            'cross_correlation_histogram': ccg_cor,
            'shift_predictor': shift_p,
            'bin_edges': bin_edges
        }
        return self.result

    def plot(self, axes=None, form='bar', **kwargs):
        if self.result is None:
            self.logger.warning("Please process the data first!")
            return
        if axes is None:
            fig, axes = plt.subplots()
        plot_th(self.result['cross_correlation_histogram'], self.result['bin_edges'], axes, form, **kwargs)
        kwargs['form'] = form
        self.set_params_plot(kwargs)
        return axes


class JointPSTHAnalyzer(Analyzer):
    ana_type = "plot_joint_psth"
    description = "Joint PSTH matrix shows the correlations of the two neurons around the reference events."
    name = "Joint Peri-Stimulus Time Histogram"

    def __init__(self, target: np.ndarray, refer: np.ndarray, event_train: np.ndarray, logger_name=None):
        super().__init__(logger_name=logger_name)
        self.target = Neuron(target)
        self.refer = Neuron(refer)
        self.event_train = event_train
        self.result = None

    def _shift_predictor(self, shift_predictor=None):
        trial_n, bin_n = self.target_bin_epochs.shape
        if shift_predictor is None:
            shift_predictor = [0]
        elif shift_predictor == 'random':
            shift_predictor = [0, np.random.randint(1, trial_n)]
        elif shift_predictor == 'average':
            shift_predictor = list(range(trial_n))
        else:
            raise ValueError(f"shift_predictor={shift_predictor} is not allowed")
        return shift_predictor

    def co_matrix(self, refer_bin_epochs):
        trial_n, bin_n = self.target_bin_epochs.shape
        trial_n2, bin_n2 = refer_bin_epochs.shape
        trial_n = min([trial_n, trial_n2])
        bin_n = min([bin_n, bin_n2])
        co_matrix = np.zeros((bin_n, bin_n))
        for i in range(trial_n):
            ta = self.target_bin_epochs[i]
            re = refer_bin_epochs[i]
            a = np.array([re for _ in re])
            for j in range(bin_n):
                a[j][a[j] > ta[j]] = ta[j]
            co_matrix += a
        return np.flipud(co_matrix)

    def process(self, shift_predictor=None, bias_start=None, bias_stop=None, bin_size=None,
                t_start=None, t_stop=None, smooth=None, kernel_size=None):
        if t_start is None:
            t_start = self.target.t_start
        if t_stop is None:
            t_stop = self.target.t_stop
        if bias_start is None:
            bias_start = 0.2
        if bias_stop is None:
            bias_stop = 0.2
        if bin_size is None:
            bin_size = 0.005

        self.set_params_data({
            'shift_predictor': shift_predictor,
            't_start': t_start,
            't_stop': t_stop,
            'bias_start': bias_start,
            'bias_stop': bias_stop,
            'bin_size': bin_size,
            'smooth': smooth,
            'kernel_size': kernel_size
        })
        self.target.cut(t_start, t_stop, True)
        t = self.target.split(self.event_train, bias_start, bias_stop, bin_size,
                              smooth=smooth, kernel_size=kernel_size)
        r = self.refer.split(self.event_train, bias_start, bias_stop, bin_size,
                             smooth=smooth, kernel_size=kernel_size)
        self.target_epochs, self.bin_edges, self.target_bin_epochs = t
        self.refer_epochs, self.bin_edges, self.refer_bin_epochs = r
        shift_predictor = self._shift_predictor(shift_predictor)
        trial_n, bin_n = self.target_bin_epochs.shape
        com_list = []
        for shp in shift_predictor:
            com_list.append(self.co_matrix(np.roll(self.refer_bin_epochs, shp)))
        com_arr = np.array(com_list)
        shift_p = np.zeros((bin_n, bin_n))
        if len(shift_predictor) > 1:
            shift_p = np.mean(com_arr, axis=0)
        com_cor = com_arr[0] - shift_p
        self.result = {
            'joint_psth_matrix': com_cor,
            'shift_predictor': shift_p
        }
        return self.result

    def plot(self, axes=None, cmap='RdBu_r'):
        import seaborn as sns
        if self.result is None:
            self.logger.warning("Please process the data first!")
            return
        if axes is None:
            fig, axes = plt.subplots()
        jps_matrix = self.result['joint_psth_matrix']
        sns.heatmap(jps_matrix, ax=axes, vmin=np.min(jps_matrix),
                    vmax=np.max(jps_matrix), cmap=cmap)
        return axes

    def target_edge_plot(self, axes=None, **kwargs):
        if self.result is None:
            self.logger.warning("Please process the data first!")
            return
        if axes is None:
            fig, axes = plt.subplots()
        jps_matrix = self.result['joint_psth_matrix']
        trial_n, bin_n = self.target_bin_epochs.shape
        plt.barh(range(bin_n), np.sum(jps_matrix, axis=1),
                 height=1, axes=axes, **kwargs)

    def refer_edge_plot(self, axes=None, **kwargs):
        if self.result is None:
            self.logger.warning("Please process the data first!")
            return
        if axes is None:
            fig, axes = plt.subplots()
        jps_matrix = self.result['joint_psth_matrix']
        trial_n, bin_n = self.target_bin_epochs.shape
        plt.bar(range(bin_n), np.sum(jps_matrix, axis=0),
                height=1, axes=axes, **kwargs)

    def main_diagonal_plot(self, axes=None, **kwargs):
        if self.result is None:
            self.logger.warning("Please process the data first!")
            return
        if axes is None:
            fig, axes = plt.subplots()
        jps_matrix = self.result['joint_psth_matrix']
        trial_n, bin_n = self.target_bin_epochs.shape
        plt.bar(range(bin_n), np.diag(jps_matrix), height=1, axes=axes, **kwargs)

    def count_diagonal_plot(self, axes=None, **kwargs):
        if self.result is None:
            self.logger.warning("Please process the data first!")
            return
        if axes is None:
            fig, axes = plt.subplots()
        jps_matrix = self.result['joint_psth_matrix']
        trial_n, bin_n = self.target_bin_epochs.shape
        plt.bar(range(bin_n), np.diag(np.flipud(jps_matrix)), height=1, axes=axes, **kwargs)

    def plot_with_edge(self):
        pass

    def plot_with_diag(self):
        pass
"""


