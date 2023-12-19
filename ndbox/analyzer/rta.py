# from .ana_base import Analyzer, Neuron, NeuronList
# import numpy as np
# from .sua import RasterAnalyzer
#
#
# class RelativeAnalyzer(Analyzer):
#     """
#     Analysis of target neuron relative to `reference` stimulus, the stimulus can
#     be the target neuron firing timestamps or the refer neuron firing timestamps
#     or the event/mark/artificial-stimulus series (which can determine a series of trial).
#
#     plot_auto_correlation():
#         Autocorrelogram to describe auto-correlation of spiking activity.
#     plot_peri_stimulus_time_hist():
#         Peri-stimulus time histogram of given spike_train.
#     plot_peri_stimulus_raster():
#         Peri-stimulus raster of given spike_train.
#     plot_cross_correlation():
#         The cross-correlation histogram of target spike train and
#         refer spike train (without consideration of trials and shift-predictor).
#
#     See Also
#     ------
#     TrialRelativeAnalyzer.plot_joint_psth()
#     TrialRelativeAnalyzer.plot_cross_correlation()
#     """
#
#     def __init__(self, spike_train, axes=None):
#         super().__init__(axes)
#         self.target = Neuron(spike_train)
#         self.bin_epochs = None
#         self.bin_edges = None
#         self.epochs = None
#         self.kwargs = None
#
#     def calculate(self, stimulus):
#         """
#         Calculate the target spiking average activity around the stimulus.
#
#         Parameters
#         ------
#         t_start: float
#             The sample beginning timestamp.
#         t_stop: float
#             The sample finishing timestamp.
#         stimulus: np.ndarray
#             1-D array, represents the stimulus timestamp. Generally, it means
#             the timestamps you are interested in.
#         bias_start: float
#             The timestamp beginning the record. Default 0.5(sec)
#         bias_stop: float
#             The timestamp finishing the record. Default 1.5(sec)
#         bin_size: float
#             The bin_size when sampling the epochs
#         """
#         self.t_start = self.kwargs.pop('t_start', self.target.t_start)
#         self.t_stop = self.kwargs.pop('t_stop', self.target.t_stop)
#         self.bias_start = self.kwargs.pop('bias_start')
#         self.bias_stop = self.kwargs.pop('bias_stop')
#         self.bin_size = self.kwargs.pop('bin_size', None)
#         smooth = self.kwargs.pop('smooth', None)
#         kernel_size = self.kwargs.pop('kernel_size', None)
#
#         self.target.cut(self.t_start, self.t_stop, True)
#         res = self.target.split(stimulus, self.bias_start, self.bias_stop, self.bin_size,
#                                 smooth=smooth, kernel_size=kernel_size)
#         if self.bin_size is None:
#             self.epochs = res
#         else:
#             self.epochs, self.bin_edges, self.bin_epochs = res
#
#     def plot_auto_correlation(self, bias_start=None, bias_stop=None, bin_size=None, **kwargs):
#         self.kwargs = kwargs
#         self.kwargs['bin_size'] = 0.005 if bin_size is None else bin_size
#         self.kwargs['bias_start'] = 0.2 if bias_start is None else bias_start
#         self.kwargs['bias_stop'] = 0.2 if bias_stop is None else bias_stop
#         self.calculate(self.target.spike_train)
#         form = self.kwargs.pop('form', 'step')
#         hist = np.average(self.bin_epochs, axis=0)
#         self.clear()
#         self.plot_th(hist, self.bin_edges, None, form, **self.kwargs)
#
#     def plot_peri_stimulus_time_hist(self, event_train, bias_start=None, bias_stop=None, bin_size=None,
#                                      **kwargs):
#         self.kwargs = kwargs
#         self.kwargs['bin_size'] = 0.005 if bin_size is None else bin_size
#         self.kwargs['bias_start'] = 0.2 if bias_start is None else bias_start
#         self.kwargs['bias_stop'] = 0.2 if bias_stop is None else bias_stop
#         self.calculate(event_train)
#         form = self.kwargs.pop('form', 'step')
#         hist = np.average(self.bin_epochs, axis=0)
#         self.clear()
#         self.plot_th(hist, self.bin_edges, None, form, **self.kwargs)
#
#     def plot_cross_correlation(self, refer, bias_start=None, bias_stop=None, bin_size=None,
#                                **kwargs):
#         self.kwargs = kwargs
#         self.kwargs['bin_size'] = 0.005 if bin_size is None else bin_size
#         self.kwargs['bias_start'] = 0.2 if bias_start is None else bias_start
#         self.kwargs['bias_stop'] = 0.2 if bias_stop is None else bias_stop
#         self.calculate(refer)
#         form = self.kwargs.pop('form', 'step')
#         hist = np.average(self.bin_epochs, axis=0)
#         self.clear()
#         self.plot_th(hist, self.bin_edges, None, form, **self.kwargs)
#
#     def plot_peri_stimulus_raster(self, event_train, bias_start=None, bias_stop=None,
#                                   **kwargs):
#         self.kwargs = kwargs
#         self.kwargs['bias_start'] = 0.2 if bias_start is None else bias_start
#         self.kwargs['bias_stop'] = 0.2 if bias_stop is None else bias_stop
#         self.calculate(event_train)
#         for i in range(len(self.epochs)):
#             if self.epochs[i].size > 0:
#                 self.epochs[i] -= event_train[i]
#         self.clear()
#         RasterAnalyzer(self.epochs, axes=self.current_axes).plot(**self.kwargs)
#
#
# class TrialRelativeAnalyzer(Analyzer):
#     """
#     Relative analysis of `target` neuron relative to `reference` neuron with consideration of
#     `stimulus`(or we say event/mark/artificial-stimulus).
#
#     Notes
#     ------
#     The `reference` can be trial aligned, and here the shift-predictor
#     can be significant.
#
#     Shift-Predictor:
#     If you do this simultaneously in both cells -- which is usually the whole idea of the experiment -- what
#     you are doing is simultaneously increasing the firing rate of both cells at the same time; thus, you've
#     introduced a relationship between the firing probabilities of the cells, just by co-stimulating them.
#
#     plot_cross_correlation():
#         The cross-correlation histogram of target spike train and refer spike train.
#     plot_joint_psth():
#             Joint peri-stimulus time histogram of target spike train and refer spike train.
#     """
#     def __init__(self, target, refer, event_train, axes=None):
#         """
#         Parameters
#         ------
#         target: np.ndarray
#             The target spike train
#         refer: np.ndarray
#             The refer spike train
#         event_train: np.ndarray
#             The event train, can be used to determine trials.
#         """
#         super().__init__(axes)
#         self.target = Neuron(target)
#         self.refer = Neuron(refer)
#         self.event_train = event_train
#         self.bias_start = None
#         self.bias_stop = None
#         self.target_epochs = None
#         self.target_bin_epochs = None
#         self.refer_epochs = None
#         self.refer_bin_epochs = None
#         self.bin_edges = None
#         self.kwargs = None
#
#     def calculate(self, bias_start=None, bias_stop=None, bin_size=None, **kwargs):
#         self.kwargs = kwargs
#         self.t_start = self.kwargs.pop('t_start', self.target.t_start)
#         self.t_stop = self.kwargs.pop('t_stop', self.target.t_stop)
#         self.bias_start = 0.2 if bias_start is None else bias_start
#         self.bias_stop = 0.2 if bias_stop is None else bias_stop
#         self.bin_size = 0.005 if bin_size is None else bin_size
#         self.smooth = self.kwargs.pop('smooth', None)
#         self.kernel_size = self.kwargs.pop('kernel_size', None)
#
#         self.target.cut(self.t_start, self.t_stop, True)
#         t = self.target.split(self.event_train, self.bias_start, self.bias_stop, self.bin_size,
#                               smooth=self.smooth, kernel_size=self.kernel_size)
#         r = self.refer.split(self.event_train, self.bias_start, self.bias_stop, self.bin_size, **self.kwargs,
#                              smooth=self.smooth, kernel_size=self.kernel_size)
#         self.target_epochs, self.bin_edges, self.target_bin_epochs = t
#         self.refer_epochs, self.bin_edges, self.refer_bin_epochs = r
#         self.shift_predictor = None
#
#     def co_matrix(self, refer_bin_epochs):
#         trial_n, bin_n = self.target_bin_epochs.shape
#         trial_n2, bin_n2 = refer_bin_epochs.shape
#         trial_n = min([trial_n, trial_n2])
#         bin_n = min([bin_n, bin_n2])
#         co_matrix = np.zeros((bin_n, bin_n))
#         for i in range(trial_n):
#             ta = self.target_bin_epochs[i]
#             re = refer_bin_epochs[i]
#             a = np.array([re for _ in re])
#             for j in range(bin_n):
#                 a[j][a[j] > ta[j]] = ta[j]
#             co_matrix += a
#         return np.flipud(co_matrix)
#
#     def _shift_predictor(self, shift_predictor=None):
#         trial_n, bin_n = self.target_bin_epochs.shape
#         if shift_predictor is None:
#             shift_predictor = [0]
#         elif shift_predictor == 'random':
#             shift_predictor = [0, np.random.randint(1, trial_n)]
#         elif shift_predictor == 'average':
#             shift_predictor = list(range(trial_n))
#         else:
#             raise ValueError(f"shift_predictor={shift_predictor} is not allowed")
#         return shift_predictor
#
#     def plot_joint_psth(self, shift_predictor=None):
#         trial_n, bin_n = self.target_bin_epochs.shape
#         if self.shift_predictor is None:
#             self.shift_predictor = self._shift_predictor(shift_predictor)
#         com_list = []
#         for shp in self.shift_predictor:
#             com_list.append(self.co_matrix(np.roll(self.refer_bin_epochs, shp)))
#         com_arr = np.array(com_list)
#         shift_p = np.zeros((bin_n, bin_n))
#         if len(self.shift_predictor) > 1:
#             shift_p = np.mean(com_arr, axis=0)
#         com_cor = com_arr[0] - shift_p
#         # now plot the matrix
#         import seaborn as sns
#         sns.heatmap(com_cor, ax=self.current_axes, vmin=np.min(com_cor),
#                     vmax=np.max(com_cor), cmap='RdBu_r')
#         self.set_spines_hide()
#         return com_cor, shift_p
#
#     def plot_cross_correlation(self, shift_predictor=None):
#         trial_n, bin_n = self.refer_bin_epochs.shape
#         self.form = self.kwargs.pop('form', 'step')
#         if self.shift_predictor is None:
#             self.shift_predictor = self._shift_predictor(shift_predictor)
#         if self.refer_bin_epochs is None:
#             self.logger.warning("Please calculate() the epochs first")
#             return
#         ccg_list = []
#         bin_edges = []
#         for shp in self.shift_predictor:
#             e_shp = np.roll(self.event_train, -shp) - self.event_train
#             ref_train = []
#             for i in range(trial_n):
#                 if self.refer_epochs[i].size > 0:
#                     ref_train.append(self.refer_epochs[i]+e_shp[i])
#             ref_train = np.concatenate(ref_train)
#             refer_epochs, bin_edges, refer_bin_epochs = \
#                 self.target.split(ref_train, self.bias_start, self.bias_stop, self.bin_size,
#                                   smooth=self.smooth, kernel_size=self.kernel_size)
#             ccg_list.append(np.average(refer_bin_epochs, axis=0))
#         shift_p = np.zeros(bin_n)
#         if len(self.shift_predictor) > 1:
#             shift_p = np.mean(np.array(ccg_list), axis=0)
#         ccg_cor = ccg_list[0] - shift_p
#         self.clear()
#         self.plot_th(ccg_cor, bin_edges, None, form=self.form, **self.kwargs)
#         return ccg_cor, shift_p
#
#
#
