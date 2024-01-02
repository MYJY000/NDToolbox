from scipy.optimize import curve_fit
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from .ana_base import Analyzer, Neuron, plot_th


def cos_tuning_model(x, m, pd, b0):
    return m * np.cos(x - pd) + b0

def prefer_direction(m, pd, b0):
    if m > 0:
        return pd
    else:
        return pd + np.pi

def cos_fit(x, y):
    fmn = y.min()
    fmx = y.max()
    # noinspection PyTupleAssignmentBalance
    params, _ = curve_fit(cos_tuning_model, x, y,
                          p0=[(fmn - fmx) / 2, 1, fmn + 1],
                          bounds=([-np.inf, 0, -np.inf], [np.inf, 2 * np.pi, np.inf]))
    return params

def metric_r2(y, y_hat):
    y_mean = np.mean(y)
    se = np.sum((y_hat - y) ** 2)
    me = np.sum((y_mean - y) ** 2)
    return 1 - se / me

class TuningAnalyzer(Analyzer):
    ana_type = "plot_tuning_curve"
    description = "Motor cortex neurons' firing rates is tuned by body part movement direction" \
                  "like hands or finger. \nIf there exists kinematics data, this kind of analysis" \
                  "can be of great help."
    name = "Kinematics Tuning Analysis"

    def __init__(self, spike_train: np.ndarray, kinematics, kin_resolution, logger_name=None):
        """
        Parameters
        ------
        kinematics: np.ndarray
            It's a 2-D array which has 2 column, the first column represents X-axis position while
            the second column represents the second column
        kin_resolution: float
            The resolution of kinematics data
         So here we know, X[i] means that at the time `i*kin_resolution` the body part's X coordinate is X[i].
         Note that you should make sure kinematics' time stamp is aligned with the spike train.
        """
        super().__init__(logger_name=logger_name)
        self.data = Neuron(spike_train)
        self.kinematics = kinematics
        self.kin_resolution = kin_resolution
        self.result = None

    def sample_direction(self, bin_size):
        pos_x = self.kinematics[:, 0]
        pos_y = self.kinematics[:, 1]
        sample_factor = int(bin_size / self.kin_resolution)
        pos_x_sampled = pos_x[::sample_factor]
        pos_y_sampled = pos_y[::sample_factor]
        pox_diff = np.diff(pos_x_sampled)
        poy_diff = np.diff(pos_y_sampled)
        theta = np.arctan(poy_diff / pox_diff)
        theta[pox_diff < 0] += np.pi
        theta[theta < 0] += 2 * np.pi
        return theta

    def split_n_dir(self, firing_rates, direction, n_dir):
        delta = 2 * np.pi / n_dir
        angle = np.zeros(n_dir + 1)
        x = []
        y = []
        for i in range(n_dir):
            angle[i + 1] = delta * (i + 1)
            epoch_mask = (direction >= angle[i]) & (direction < angle[i + 1])
            theta_epoch = direction[epoch_mask]
            fr_epoch = firing_rates[epoch_mask]
            x.append(theta_epoch.mean())
            y.append(fr_epoch.mean())
        return angle, np.array(x), np.array(y)

    # noinspection PyIncorrectDocstring
    def process(self, delay=None, n_dir=None, bin_size=None, smooth=None, kernel_size=None):
        """
        Parameters
        ------
        n_dir: int
            Divide [0, 2*pi] into `n_dir` parts, so the angle and the firing rates will be
            averaged within each part.
        bin_size: float
            Usually, the raw kinematics data and firing data should be re-sampled to bigger bin to make
            the data not so sparse. Also, here we also need smooth the firing data to get a better fit.
        delay: float
            Time delay of firing stamp according with kinematics data, it can be positive or negative.
        """
        if delay is None:
            delay = 0
        if n_dir is None:
            n_dir = 8
        if bin_size is None:
            bin_size = 0.5
        if smooth is None:
            smooth = 'gas'
        if kernel_size is None:
            kernel_size = 10
        self.set_params_data({
            'delay': delay,
            'n_dir': n_dir,
            'bin_size': bin_size,
            'smooth': smooth,
            'kernel_size': kernel_size
        })

        sp = self.data.spike_train
        sp += delay
        self.data = Neuron(sp)
        direction = self.sample_direction(bin_size)
        self.logger.info("Position data transferred to direction data success.")
        firing_rates, bin_edges = self.data.sample(bin_size, smooth, kernel_size)
        data_length = min(direction.size, firing_rates.size)
        direction = direction[:data_length]
        firing_rates = firing_rates[:data_length]
        fl = ~np.isnan(direction)
        direction = direction[fl]
        firing_rates = firing_rates[fl]
        self.logger.info("Firing rate time aligned with direction data success!.")
        angle_edges, dir_fit, fir_fit = self.split_n_dir(firing_rates, direction, n_dir)
        self.logger.info("Data split finished, start fitting the firing rates with the direction.")
        params = cos_fit(dir_fit, fir_fit)
        self.logger.info(f"Cosine tuning fit success, params is {params}")
        fir_hat = cos_tuning_model(dir_fit, *params)
        err = np.abs(fir_fit-fir_hat)
        R2 = metric_r2(fir_fit, fir_hat)
        self.logger.info(f"R2 is {R2}")
        PD = prefer_direction(*params)
        self.logger.info(f"Preferred direction is {PD}")
        self.result = {
            'direction': direction,
            'firing_rates': firing_rates,
            'dir_fit': dir_fit,
            'fir_fit': fir_fit,
            'angle_edges': angle_edges,
            'params': params,
            'R2': R2,
            'PD': PD,
            'err': err
        }
        return self.result

    def plot(self, axes=None, **kwargs):
        if self.result is None:
            self.logger.warning("Please process the data first!")
            return
        if axes is None:
            fig, axes = plt.subplots()

        sc = kwargs.pop('scatter', False)
        if isinstance(sc, dict):
            axes.scatter(self.result['dir_fit'], self.result['fir_fit'], **sc)
        if isinstance(sc, bool):
            if sc:
                axes.scatter(self.result['dir_fit'], self.result['fir_fit'], s=8)

        er = kwargs.pop('error_bar', False)
        erp = cos_tuning_model(self.result['dir_fit'], *self.result['params'])
        if isinstance(er, dict):
            axes.errorbar(self.result['dir_fit'], erp, self.result['err'], **er)
        if isinstance(er, bool):
            if er:
                axes.errorbar(self.result['dir_fit'], erp, self.result['err'],
                              fmt='o', ecolor='r', elinewidth=1, capsize=3,
                              capthick=2, barsabove=True)

        x_fit = np.linspace(0, 2 * np.pi, 1000)
        y_fit = cos_tuning_model(x_fit, *self.result['params'])
        axes.plot(x_fit, y_fit, **kwargs)
        axes.set_xticks([0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi])
        axes.set_xticklabels(['0', r'$\frac{\pi}{2}$', r'$\pi$', r'$\frac{3}{2}\pi$', r'$2\pi$'])
        return axes


class PeriStimulusAnalyzer(Analyzer):
    ana_type = "plot_peri_stimulus_time_histogram"
    description = "Peri-stimulus time histogram shows the conditional probability of a spike at time `t0+t` on " \
                  "the condition that there is a reference event at time `t0`."
    name = "Peri-Stimulus Time Histogram"

    def __init__(self, spike_train: np.ndarray, event_train: np.ndarray,  logger_name=None):
        super().__init__(logger_name=logger_name)
        self.data = Neuron(spike_train)
        self.event_train = event_train
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
        epochs, bin_edges, bin_epochs = self.data.split(self.event_train, bias_start, bias_stop,
                                                        bin_size, smooth=smooth, kernel_size=kernel_size)
        self.result = {
            'peri_stimulus_raster': [epochs[i]-self.event_train[i] for i in range(self.event_train.size)],
            'peri_stimulus_time_hist': np.average(bin_epochs, axis=0),
            'bin_edges': bin_edges
        }
        return self.result

    def plot(self, axes=None, form='bar', **kwargs):
        if self.result is None:
            self.logger.warning("Please process the data first!")
            return
        if axes is None:
            fig, axes = plt.subplots()
        plot_th(self.result['peri_stimulus_time_hist'], self.result['bin_edges'], axes, form, **kwargs)
        kwargs['form'] = form
        self.set_params_plot(kwargs)
        return axes

    def raster(self, axes=None, hide_spine=None, **kwargs):
        if self.result is None:
            self.logger.warning("Please process the data first!")
            return
        if axes is None:
            fig, axes = plt.subplots()
        if hide_spine is None:
            hide_spine = True
        linelengths = kwargs.pop("linelengths", 0.86)
        linewidths = kwargs.pop("linewidths", 0.8)
        axes.eventplot(self.result['peri_stimulus_raster'],
                       linelengths=linelengths, linewidths=linewidths, **kwargs)
        if hide_spine:
            axes.spines['top'].set_visible(False)
            axes.spines['bottom'].set_visible(False)
            axes.spines['left'].set_visible(False)
            axes.spines['right'].set_visible(False)
            axes.set_xticks([])
            axes.set_yticks([])
        return axes




