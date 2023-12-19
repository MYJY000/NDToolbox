from .ana_base import Analyzer, NeuronList, plot_th
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from typing import List

class RasterMulAnalyzer(Analyzer):
    ana_type = 'plot_raster'
    description = 'Raster plot of multiple neurons.'
    name = "Raster Plot Analysis"

    def __init__(self, spike_train: List[np.ndarray], logger_name=None):
        super().__init__(logger_name=logger_name)
        self.data = NeuronList(spike_train)
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

    def plot(self, axes=None, hide_spines=None, **kwargs):
        if self.result is None:
            self.logger.warning("Please process the data first!")
            return
        if axes is None:
            fig, axes = plt.subplots()
        if hide_spines is None:
            hide_spines = False
        linelengths = kwargs.pop("linelengths", 0.86)
        linewidths = kwargs.pop("linewidths", 0.8)
        axes.eventplot(self.data.spike_train, linelengths=linelengths,
                       linewidths=linewidths, **kwargs)
        if hide_spines:
            axes.spines['top'].set_visible(False)
            axes.spines['bottom'].set_visible(False)
            axes.spines['left'].set_visible(False)
            axes.spines['right'].set_visible(False)
            axes.set_xticks([])
            axes.set_yticks([])
        kwargs["linelengths"] = linelengths
        kwargs["linewidths"] = linewidths
        kwargs["hide_spine"] = hide_spines
        self.set_params_plot(kwargs)
        return axes






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
