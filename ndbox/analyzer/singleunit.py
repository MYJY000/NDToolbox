from .basic import Analysis
from ndbox.dataset import NeuralDataset
from ndbox.dataset.utils import cut, sample, split
import numpy as np
import matplotlib.pyplot as plt
from .model import CosineTuningModel, cos_tuning_model
import os

class TimeHistogram(Analysis):
    """plot the time histogram for each spike-train given `dataset`(a series of spiketrains)
    """
    def __init__(self, dataset: NeuralDataset):
        """
        `dataset`: contains a series of spiketrains.
        """
        super(TimeHistogram, self).__init__(dataset)

    def analyze(self, bin_size, t_start=None, t_stop=None, smooth=True, window=0.8):
        if t_start is None:
            t_start = self.dataset.get_t_start()
        if t_stop is None:
            t_stop = self.dataset.get_t_stop()
            
        firing_rates = self.dataset.sample_spiketrains(bin_size, t_start, t_stop, smooth, window)
        self.params_data = {
            't_start': t_start,
            't_stop': t_stop,
            'bin_size': bin_size,
            'smooth': smooth,
            'window': window
        }
        self.anares = {'firing_rates': firing_rates/bin_size}
        return self.anares

    def plot(self, root, form='bar', **kwargs):
        """
        root: the saving root of result png files
        form: 'bar' or 'curve'
        """
        t_steps, n_units = self.anares['firing_rates'].shape
        x = np.linspace(self.params_data['t_start'], self.params_data['t_stop'], t_steps+1)
        x = x[:-1] + self.params_data['bin_size']/2
        
        fig = plt.figure()
        for j in range(n_units):
            axes = fig.add_subplot(1,1,1)
            axes.set_xlabel('time (s)')
            axes.set_ylabel(r'firing rates ($s^{-1}$)')
            if form == 'bar':
                axes.bar(x, self.anares['firing_rates'][:, j], width=self.params_data['bin_size'], **kwargs)
            elif form == 'curve':
                axes.plot(x, self.anares['firing_rates'][:, j], **kwargs)
            fig.savefig(os.path.join(root, str(j)+'.png'), transparent=True, dpi=200, pad_inches=0)
            plt.clf()

class CumulativeActivity(Analysis):
    """plot the cumulative activity(when a spike found, the curve step up) for each spike-train given `dataset`(a series of spiketrains)
    """
    def __init__(self, dataset: NeuralDataset):
        """
        `dataset`: contains a series of spiketrains.
        """
        super(CumulativeActivity, self).__init__(dataset)

    def analyze(self, bin_size, t_start=None, t_stop=None):
        if t_start is None:
            t_start = self.dataset.get_t_start()
        if t_stop is None:
            t_stop = self.dataset.get_t_stop()
        # firing_rates(t, n)
        firing_rates = self.dataset.sample_spiketrains(bin_size, t_start, t_stop, False)
        self.params_data = {
            't_start': t_start,
            't_stop': t_stop,
            'bin_size': bin_size,
        }
        self.anares = {'cum_activity': np.cumsum(firing_rates, axis=0)}
        return self.anares

    def plot(self, root, **kwargs):
        t_steps, n_units = self.anares['cum_activity'].shape
        x = np.linspace(self.params_data['t_start'], self.params_data['t_stop'], t_steps+1)
        x = x[:-1] + self.params_data['bin_size']/2

        fig = plt.figure()
        for j in range(n_units):
            axes = fig.add_subplot(1,1,1)
            axes.set_xlabel('time (s)')
            axes.set_ylabel('spike counts')
            axes.step(x, self.anares['cum_activity'][:, j], **kwargs)
            fig.savefig(os.path.join(root, str(j)+'.png'), transparent=True, dpi=200, pad_inches=0)
            plt.clf()

class Rastergram(Analysis):
    """raster plot for all spiketrains given `dataset`(maybe a indices is needed)
    """
    def __init__(self, dataset: NeuralDataset):
        """
        `dataset`: contains a series of spiketrains.
        """
        super(Rastergram, self).__init__(dataset)

    def analyze(self, t_start=None, t_stop=None, indices: list[int]=None):
        if t_start is None:
            t_start = self.dataset.get_t_start()
        if t_stop is None:
            t_stop = self.dataset.get_t_stop()
        
        spike_trains = self.dataset.spiketrains
        spd = [spike_trains[idx] for idx in indices]
        spd = cut(spd, t_start, t_stop)

        self.params_data = {
            't_start': t_start,
            't_stop': t_stop,
            'indices': indices
        }
        self.anares = {'spike_trains': spd}
        return self.anares

    def plot(self, root, **kwargs):
        fig = plt.figure()
        axes = fig.add_subplot(1,1,1)
        linelengths = kwargs.pop("linelengths", 0.86)
        linewidths = kwargs.pop("linewidths", 0.8)
        axes.set_yticks(list(range(len(self.params_data['indices']))), self.params_data['indices'])
        axes.set_xlabel('time (s)')
        axes.set_ylabel('unit index')
        axes.eventplot(self.anares['spike_trains'], linelengths=linelengths, 
                       linewidths=linewidths, **kwargs)
        fig.savefig(os.path.join(root, 'rasters.png'), transparent=True, dpi=200, pad_inches=0)
        plt.clf()
        return axes

class ISIDistribution(Analysis):
    """'plot the inter spike interval distribution of spike train.
    """
    def __init__(self, dataset: NeuralDataset):
        """
        `dataset`: contains a series of spiketrains.
        """
        super(ISIDistribution, self).__init__(dataset)

    def analyze(self, t_start=None, t_stop=None, min_width=None, max_width=None, bin_size=None):
        if t_start is None:
            t_start = self.dataset.get_t_start()
        if t_stop is None:
            t_stop = self.dataset.get_t_stop()
        if bin_size is None:
            bin_size = 0.01
        if min_width is None:
            min_width = 0
        if max_width is None:
            max_width = 0.8
        
        spiketrains = self.dataset.spiketrains
        spiketrains = cut(spiketrains, t_start, t_stop)
        intervals = []
        for sp in spiketrains:
            intervals.append(np.diff(sp))
        isi_dis = sample(intervals, bin_size, min_width, max_width)

        self.params_data = {
            't_start': t_start,
            't_stop': t_stop,
            'bin_size': bin_size,
            'min_width': min_width,
            'max_width': max_width
        }
        self.anares = {'isi_dis': isi_dis}
        return self.anares

    def plot(self, root, **kwargs):
        fig = plt.figure()
        t_steps, n_units = self.anares['isi_dis'].shape
        x = np.linspace(self.params_data['min_width'], self.params_data['max_width'], t_steps+1)
        x = x[:-1] + self.params_data['bin_size']/2
        for j in range(n_units):
            axes = fig.add_subplot(1,1,1)
            axes.set_xlabel('time interval (s)')
            axes.set_ylabel('interval distribution')
            axes.bar(x, self.anares['isi_dis'][:, j], width=self.params_data['bin_size'], **kwargs)
            fig.savefig(os.path.join(root, str(j)+'.png'), transparent=True, dpi=200, pad_inches=0)
            plt.clf()

class ISITimeHistogram(Analysis):
    """plot the inter spike interval time histogram of spike train.
    """
    def __init__(self, dataset: NeuralDataset):
        """
        `dataset`: contains a series of spiketrains.
        """
        super(ISITimeHistogram, self).__init__(dataset)

    def analyze(self, t_start=None, t_stop=None):
        if t_start is None:
            t_start = self.dataset.get_t_start()
        if t_stop is None:
            t_stop = self.dataset.get_t_stop()
        
        spiketrains = self.dataset.spiketrains
        spiketrains = cut(spiketrains, t_start, t_stop)
        intervals = []
        for sp in spiketrains:
            intervals.append(np.diff(sp))
        self.params_data = {
            't_start': t_start,
            't_stop': t_stop
        }
        self.anares = {'spiketrains': spiketrains, 'intervals': intervals}
        return self.anares

    def plot(self, root, **kwargs):
        fig = plt.figure()
        n_units = len(self.anares['spiketrains'])
        for j in range(n_units):
            axes = fig.add_subplot(1,1,1)
            axes.set_xlabel('time (s)')
            axes.set_ylabel('interval (s)')
            xtick = self.anares['spiketrains'][j][:-1]
            y = self.anares['intervals'][j]
            axes.vlines(xtick, np.zeros(y.size), y, **kwargs)
            fig.savefig(os.path.join(root, str(j)+'.png'), transparent=True, dpi=200, pad_inches=0)
            plt.clf()

class InstantFrequecy(Analysis):
    """"plot the instantaneous frequency of spike train.
    """
    def __init__(self, dataset: NeuralDataset):
        """
        `dataset`: contains a series of spiketrains.
        """
        super(InstantFrequecy, self).__init__(dataset)

    def analyze(self, t_start=None, t_stop=None, max_freq=1000):
        if t_start is None:
            t_start = self.dataset.get_t_start()
        if t_stop is None:
            t_stop = self.dataset.get_t_stop()
        
        spiketrains = self.dataset.spiketrains
        spiketrains = cut(spiketrains, t_start, t_stop)
        freqs = []
        for sp in spiketrains:
            iv = np.diff(sp)
            iv[iv < 1/max_freq] = 1/max_freq
            freqs.append(1/iv)
        self.params_data = {
            't_start': t_start,
            't_stop': t_stop,
            'max_freq': max_freq
        }
        self.anares = {'spiketrains': spiketrains, 'freqs': freqs}
        return self.anares

    def plot(self, root, **kwargs):
        fig = plt.figure()
        n_units = len(self.anares['spiketrains'])
        for j in range(n_units):
            axes = fig.add_subplot(1,1,1)
            axes.set_xlabel('time (s)')
            axes.set_ylabel(r'instantaneous freqeuncy ($s-1$)')
            xtick = self.anares['spiketrains'][j][:-1]
            y = self.anares['freqs'][j]
            axes.vlines(xtick, np.zeros(y.size), y, **kwargs)
            fig.savefig(os.path.join(root, str(j)+'.png'), transparent=True, dpi=200, pad_inches=0)
            plt.clf()

class PoincareMap(Analysis):
    """poincare map of spike train, which is a scatter plot represents the relation between adjacent intervals
    """
    def __init__(self, dataset: NeuralDataset):
        """
        `dataset`: contains a series of spiketrains.
        """
        super(PoincareMap, self).__init__(dataset)

    def analyze(self, t_start=None, t_stop=None):
        if t_start is None:
            t_start = self.dataset.get_t_start()
        if t_stop is None:
            t_stop = self.dataset.get_t_stop()
        
        spiketrains = self.dataset.spiketrains
        spiketrains = cut(spiketrains, t_start, t_stop)
        intervals = [np.diff(sp) for sp in spiketrains]
        self.params_data = {'t_start': t_start, 't_stop': t_stop}
        
        self.anares = {'intervals': intervals}
        return self.anares

    def plot(self, root, **kwargs):
        fig = plt.figure()
        n_units = len(self.anares['intervals'])
        for j in range(n_units):
            axes = fig.add_subplot(1,1,1)
            poincare_x = self.anares['intervals'][j][1:]
            poincare_y = self.anares['intervals'][j][:-1]
            axes.set_xlabel('interval (s) - $t_{n}$')
            axes.set_ylabel('interval (s) - $t_{n-1}$')
            axes.scatter(poincare_x, poincare_y, **kwargs)
            fig.savefig(os.path.join(root, str(j)+'.png'), transparent=True, dpi=200, pad_inches=0)
            plt.clf()

class Autocorrelogram(Analysis):
    """Autocorrelogram shows the conditional probability of a spike at time `t_0+t` on the condition that there is a spike at time `t_0`.'
    """

    def __init__(self, dataset: NeuralDataset):
        """
        `dataset`: contains a series of spiketrains.
        """
        super(Autocorrelogram, self).__init__(dataset)

    def analyze(self, t_start=None, t_stop=None, bias_start=None, bias_stop=None, bin_size=None):
        if t_start is None:
            t_start = self.dataset.get_t_start()
        if t_stop is None:
            t_stop = self.dataset.get_t_stop()
        if bias_start is None:
            bias_start = 0.5
        if bias_stop is None:
            bias_stop = 0.5
        if bin_size is None:
            bin_size = 0.01
        self.params_data = {
            't_start': t_start,
            't_stop': t_stop,
            'bias_start': bias_start,
            'bias_stop': bias_stop,
            'bin_size': bin_size
        }

        spiketrains = self.dataset.spiketrains
        autocorrelogram = []
        for spt in spiketrains:
            stimulus = spt[spt<t_stop]
            stimulus = stimulus[stimulus>=t_start]
            spepochs = split([spt], stimulus, bias_start, bias_stop, bin_size, True)
            spepochs = spepochs.squeeze()
            autocorrelogram.append(spepochs.mean(axis=1))
        self.anares = {'autocorrelogram': np.stack(autocorrelogram).T / bin_size}
        return self.anares

    def plot(self, root, form='bar', **kwargs):
        """
        root: the saving root of result png files
        form: 'bar' or 'step' or 'curve'
        """
        t_steps, n_units = self.anares['autocorrelogram'].shape
        x = np.linspace(-self.params_data['bias_start'], self.params_data['bias_stop'], t_steps+1)
        x = x[:-1] + self.params_data['bin_size']/2
        
        fig = plt.figure()
        for j in range(n_units):
            axes = fig.add_subplot(1,1,1)
            axes.set_xlabel('time (s)')
            axes.set_ylabel(r'firing rates ($s^{-1}$)')
            if form == 'bar':
                axes.bar(x, self.anares['autocorrelogram'][:, j], width=self.params_data['bin_size'], **kwargs)
            elif form == 'curve':
                axes.plot(x, self.anares['autocorrelogram'][:, j], **kwargs)
            elif form == 'step':
                axes.step(x, self.anares['autocorrelogram'][:, j], **kwargs)
            fig.savefig(os.path.join(root, str(j)+'.png'), transparent=True, dpi=200, pad_inches=0)
            plt.clf()

class PeriStimulusTimeHistogram(Analysis):
    """Peri-stimulus time histogram shows the conditional probability of a spike at time `t0+t` on the condition that there is a reference event at time `t0`.
    """
    def __init__(self, dataset: NeuralDataset):
        super(PeriStimulusTimeHistogram, self).__init__(dataset)
    
    def analyze(self, event: str|np.ndarray, t_start: float=None, t_stop: float=None, 
                bias_start: float=None, bias_stop: float=None, bin_size: float=None) -> dict:
        """
        event: str or np.ndarray
            str(the key) if the dataset contains event time series, or ndarray if not
        """
        eventrain = self.dataset.fetch_event_stamps(event)
        if t_start is None:
            t_start = self.dataset.get_t_start()
        if t_stop is None:
            t_stop = self.dataset.get_t_stop()
        if bias_start is None:
            bias_start = 0.5
        if bias_stop is None:
            bias_stop = 0.5
        if bin_size is None:
            bin_size = 0.01
        self.params_data = {
            't_start': t_start,
            't_stop': t_stop,
            'bias_start': bias_start,
            'bias_stop': bias_stop,
            'bin_size': bin_size
        }
        spiketrains = self.dataset.spiketrains
        psth = []
        for spt in spiketrains:
            stimulus = eventrain[eventrain<t_stop]
            stimulus = stimulus[stimulus>=t_start]
            spepochs = split([spt], stimulus, bias_start, bias_stop, bin_size, True)
            spepochs = spepochs.squeeze()
            psth.append(spepochs.mean(axis=1))
        self.anares = {'psth': np.stack(psth).T / bin_size}
        return self.anares
    
    def plot(self, root, form='bar', **kwargs):
        """
        root: the saving root of result png files
        form: 'bar' or 'step' or 'curve'
        """
        t_steps, n_units = self.anares['psth'].shape
        x = np.linspace(-self.params_data['bias_start'], self.params_data['bias_stop'], t_steps+1)
        x = x[:-1] + self.params_data['bin_size']/2
        
        fig = plt.figure()
        for j in range(n_units):
            axes = fig.add_subplot(1,1,1)
            axes.set_xlabel('time (s)')
            axes.set_ylabel(r'firing rates ($s^{-1}$)')
            if form == 'bar':
                axes.bar(x, self.anares['psth'][:, j], width=self.params_data['bin_size'], **kwargs)
            elif form == 'curve':
                axes.plot(x, self.anares['psth'][:, j], **kwargs)
            elif form == 'step':
                axes.step(x, self.anares['psth'][:, j], **kwargs)
            fig.savefig(os.path.join(root, str(j)+'.png'), transparent=True, dpi=200, pad_inches=0)
            plt.clf()
    

# TuningAnalyzer
class KinematicsTuningCurve(Analysis):
    """
    A tuning curve describes the dependence of a neuron's firing on a particular variable thought 
    to be represented by the brain. A 'tuned' neuron may selectively respond to stimuli within a 
    particular band of some stimulus variable, which could be a spatial or temporal frequency, or 
    some other property of the stimulus such as its orientation, position or depth. 
    """
    def __init__(self, dataset: NeuralDataset):
        super(KinematicsTuningCurve, self).__init__(dataset)
    
    def analyze(self, behavior, bin_size, n_direction=8, 
                t_start=None, t_stop=None, smooth=True, window=0.8) -> dict:
        """
        behavior: str or np.ndarray, the kinematics data, such as hand position.
        bin_size: float, bin size when resample the firing rates and behavior data.

        """
        if t_start is None:
            t_start = self.dataset.get_t_start()
        if t_stop is None:
            t_stop = self.dataset.get_t_stop()
        firing_rates = self.dataset.sample_spiketrains(bin_size, t_start, t_stop, smooth, window)
        kinematics = self.dataset.fetch_behaviors(behavior)
        kinematics = self.dataset.resample_behaviors(behavior, bin_size, t_start, t_stop)

        cos_model = CosineTuningModel(n_direction)
        pds = []
        tuning_xs = []
        tuning_ys = []
        fit_params = []
        for j in range(firing_rates.shape[1]):
            pd = cos_model.fit(firing_rates[:, j], kinematics, bin_size)
            pds.append(pd)
            tuning_xs.append(cos_model.tuning_x)
            tuning_ys.append(cos_model.tuning_y)
            fit_params.append(cos_model.params)

        self.params_data = {
            't_start': t_start,
            't_stop': t_stop,
            'bin_size': bin_size,
            'smooth': smooth,
            'window': window
        }

        self.anares = {
            "prefer_direction": pds,
            "radian": tuning_xs,
            "firing_rate": tuning_ys,
            "cos_model_param": fit_params
        }
        return self.anares
    
    def plot(self, root, pd_aligned=False, r2=False, **kwargs) -> None:
        fig = plt.figure()
        n_units = len(self.anares["prefer_direction"])

        x_fit = np.linspace(0, 2 * np.pi, 1000)
        for j in range(n_units):
            axes = fig.add_subplot(1,1,1)
            axes.set_xlabel('radian')
            axes.set_ylabel(r'firing rates ($s^{-1}$)')
            axes.scatter(self.anares["radian"][j], self.anares["firing_rate"][j]) 
            y_fit = cos_tuning_model(x_fit, *self.anares['cos_model_param'][j])
            axes.plot(x_fit, y_fit, **kwargs)
            if pd_aligned:
                pass
            if r2:
                pass
            axes.set_xticks([0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi])
            axes.set_xticklabels(['0', r'$\frac{\pi}{2}$', r'$\pi$', r'$\frac{3}{2}\pi$', r'$2\pi$'])
            fig.savefig(os.path.join(root, str(j)+'.png'), transparent=True, dpi=200, pad_inches=0)
            plt.clf()

