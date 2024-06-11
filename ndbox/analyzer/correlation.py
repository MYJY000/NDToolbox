from .basic import Analysis
from ndbox.dataset import NeuralDataset
import numpy as np
import matplotlib.pyplot as plt
from ndbox.dataset.utils import cut, sample, split, co_matrix
import os
import seaborn as sns

class CovarianceMap(Analysis):
    def __init__(self, dataset: NeuralDataset):
        super().__init__(dataset)
    
    def analyze(self, bin_size, t_start=None, t_stop=None, smooth=True, window=0.8, indices: list[int]=None) -> dict:
        if t_start is None:
            t_start = self.dataset.get_t_start()
        if t_stop is None:
            t_stop = self.dataset.get_t_stop()
        
        firing_rates = self.dataset.sample_spiketrains(bin_size, t_start, t_stop, smooth, window)
        if indices is not None:
            firing_rates = firing_rates[:, indices]
        firing_rates = firing_rates/bin_size

        self.params_data = {
            't_start': t_start,
            't_stop': t_stop,
            'bin_size': bin_size,
            'smooth': smooth,
            'window': window,
            'indices': indices
        }
        cov_map = np.cov(firing_rates.T)
        self.anares = {'cov_map': cov_map}
        return self.anares
    
    def plot(self, path: str, **kwargs) -> None:
        fig = plt.figure()
        axes = fig.add_subplot(1,1,1)
        sns.heatmap(ax=axes, data=self.anares["cov_map"])
        fig.savefig(path, transparent=True, dpi=200, pad_inches=0)
        plt.clf()


class CorrelationCoeffMap(Analysis):
    def __init__(self, dataset: NeuralDataset):
        super().__init__(dataset)
    def analyze(self, bin_size, t_start=None, t_stop=None, smooth=True, window=0.8, indices: list[int]=None) -> dict:
        if t_start is None:
            t_start = self.dataset.get_t_start()
        if t_stop is None:
            t_stop = self.dataset.get_t_stop()
        
        firing_rates = self.dataset.sample_spiketrains(bin_size, t_start, t_stop, smooth, window)
        if indices is not None:
            firing_rates = firing_rates[:, indices]
        firing_rates = firing_rates/bin_size

        self.params_data = {
            't_start': t_start,
            't_stop': t_stop,
            'bin_size': bin_size,
            'smooth': smooth,
            'window': window,
            'indices': indices
        }
        corrcoef = np.corrcoef(firing_rates.T)
        self.anares = {'corrcoef': corrcoef}
        return self.anares
    
    def plot(self, path: str, **kwargs) -> None:
        fig = plt.figure()
        axes = fig.add_subplot(1,1,1)
        sns.heatmap(ax=axes, data=self.anares["corrcoef"])
        fig.savefig(path, transparent=True, dpi=200, pad_inches=0)
        plt.clf()
    

class STTCMap(Analysis):
    def __init__(self, dataset: NeuralDataset):
        super().__init__(dataset)
    
    def analyze(self, t_start=None, t_stop=None, slide=0.02, indices: list[int]=None) -> dict:
        if t_start is None:
            t_start = self.dataset.get_t_start()
        if t_stop is None:
            t_stop = self.dataset.get_t_stop()
        
        neuon_num = len(self.dataset.spiketrains)

        spiketrains = [self.dataset.spiketrains[i] for i in range(neuon_num) if i in indices]
        spiketrains = cut(spiketrains, t_start, t_stop)
        
        import neo
        import quantities as pq
        from elephant.spike_train_correlation import spike_time_tiling_coefficient
        neuon_num = len(spiketrains)
        sttc_map = np.zeros((neuon_num, neuon_num))
        for i in range(neuon_num):
            for j in range(i, neuon_num):
                si = neo.SpikeTrain(spiketrains[i], units='s', t_start=t_start, t_stop=t_stop)
                sj = neo.SpikeTrain(spiketrains[j], units='s', t_start=t_start, t_stop=t_stop)
                sttc = spike_time_tiling_coefficient(si, sj)
                sttc_map[i][j] = sttc
                sttc_map[j][i] = sttc

        self.params_data = {
            't_start': t_start,
            't_stop': t_stop,
            'slide': slide,
            'indices': indices
        }
        self.anares = {'sttc_map': sttc_map}
        return self.anares
    
    def plot(self, path: str, **kwargs) -> None:
        fig = plt.figure()
        axes = fig.add_subplot(1,1,1)
        sns.heatmap(ax=axes, data=self.anares["sttc_map"])
        fig.savefig(path, transparent=True, dpi=200, pad_inches=0)
        plt.clf()


class VPDMap(Analysis):
    def __init__(self, dataset: NeuralDataset):
        super().__init__(dataset)
    
    def analyze(self, t_start=None, t_stop=None, cost_factor=1.0, indices: list[int]=None) -> dict:
        if t_start is None:
            t_start = self.dataset.get_t_start()
        if t_stop is None:
            t_stop = self.dataset.get_t_stop()

        import quantities as pq
        import neo
        from elephant.spike_train_dissimilarity import victor_purpura_distance
        spiketrains = self.dataset.spiketrains
        if indices is not None:
            spiketrains = [spiketrains[i] for i in range(len(spiketrains)) if i in indices]
        spiketrains = cut(spiketrains, t_start, t_stop)
        spiketrains = [neo.SpikeTrain(sp, units='s', t_start=t_start, t_stop=t_stop) for sp in spiketrains]
        vpd = victor_purpura_distance(spiketrains, cost_factor=cost_factor*pq.Hz)
        self.params_data = {
            't_start': t_start,
            't_stop': t_stop,
            'cost_factor': cost_factor,
            'indices': indices
        }
        self.anares = {'vpd': vpd}
        return self.anares
    
    def plot(self, path: str, **kwargs) -> None:
        fig = plt.figure()
        axes = fig.add_subplot(1,1,1)
        sns.heatmap(ax=axes, data=self.anares["vpd"])
        fig.savefig(path, transparent=True, dpi=200, pad_inches=0)
        plt.clf()


class VRDMap(Analysis):
    def __init__(self, dataset: NeuralDataset):
        super().__init__(dataset)
    
    def analyze(self, t_start=None, t_stop=None, time_constant=1.0, indices: list[int]=None) -> dict:
        if t_start is None:
            t_start = self.dataset.get_t_start()
        if t_stop is None:
            t_stop = self.dataset.get_t_stop()

        import quantities as pq
        import neo
        from elephant.spike_train_dissimilarity import van_rossum_distance

        spiketrains = self.dataset.spiketrains
        if indices is not None:
            spiketrains = [spiketrains[i] for i in range(len(spiketrains)) if i in indices]
        spiketrains = cut(spiketrains, t_start, t_stop)
        spiketrains = [neo.SpikeTrain(sp, units='s', t_start=t_start, t_stop=t_stop) for sp in spiketrains]
        vrd = van_rossum_distance(spiketrains, time_constant=time_constant*pq.s)
        self.params_data = {
            't_start': t_start,
            't_stop': t_stop,
            'time_constant': time_constant,
            'indices': indices
        }
        self.anares = {'vrd': vrd}
        return self.anares
    
    def plot(self, path: str, **kwargs) -> None:
        fig = plt.figure()
        axes = fig.add_subplot(1,1,1)
        sns.heatmap(ax=axes, data=self.anares["vrd"], **kwargs)
        fig.savefig(path, transparent=True, dpi=200, pad_inches=0)
        plt.clf()


class CrossCorrelogram(Analysis):
    """
    Relative analysis of spiketrain `spt1` and `spt2` shows the conditional firing rates of `spt1` 
    at time `t0+t` on the condition that `spt2` fires at time `t0`.
    """
    def __init__(self, dataset: NeuralDataset):
        super().__init__(dataset)
    
    def analyze(self, t_start=None, t_stop=None, indices: list[int]=None, bias_start=None, bias_stop=None, bin_size=None) -> dict:
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
        spiketrains = [self.dataset.spiketrains[i] for i in range(len(self.dataset.spiketrains)) if i in indices]
        neuon_num = len(spiketrains)

        corre_histogram = []
        corre_nureonidx = []
        for i in range(neuon_num):
            for j in range(i+1, neuon_num):
                stimulus = spiketrains[j]
                stimulus = stimulus[stimulus>=t_start]
                stimulus = stimulus[stimulus<t_stop]
                spepochs = split([spiketrains[i]], stimulus, bias_start, bias_stop, bin_size, True)
                spepochs = spepochs.squeeze()
                corre_histogram.append(spepochs.mean(axis=1))
                corre_nureonidx.append([i, j])
        
        self.params_data = {
            't_start': t_start,
            't_stop': t_stop,
            'indices': indices,
            'bias_start': bias_start,
            'bias_stop': bias_stop,
            'bin_size': bin_size            
        }

        self.anares = {
            'corre_histogram': np.stack(corre_histogram).T / bin_size,
            'corre_nureonidx': np.stack(corre_nureonidx)
        }

        return self.anares

    def plot(self, root, form='bar', **kwargs):
        """
        root: the saving root of result png files
        form: 'bar' or 'step' or 'curve'
        """
        t_steps, n_units = self.anares['corre_histogram'].shape
        x = np.linspace(-self.params_data['bias_start'], self.params_data['bias_stop'], t_steps+1)
        x = x[:-1] + self.params_data['bin_size']/2
        
        fig = plt.figure()
        for j in range(n_units):
            axes = fig.add_subplot(1,1,1)
            axes.set_xlabel('time (s)')
            axes.set_ylabel(r'conditional firing rates ($s^{-1}$)')
            if form == 'bar':
                axes.bar(x, self.anares['corre_histogram'][:, j], width=self.params_data['bin_size'], **kwargs)
            elif form == 'curve':
                axes.plot(x, self.anares['corre_histogram'][:, j], **kwargs)
            elif form == 'step':
                axes.step(x, self.anares['corre_histogram'][:, j], **kwargs)
            idx = self.anares['corre_nureonidx'][j]
            file_name = str(idx[0]) + "_" + str(idx[1])
            fig.savefig(os.path.join(root, file_name+'.png'), transparent=True, dpi=200, pad_inches=0)
            plt.clf()


class JointPSTH(Analysis):
    def __init__(self, dataset: NeuralDataset):
        super().__init__(dataset)
    
    def analyze(self, eventrain, t_start=None, t_stop=None, indices: list[int]=None,
                bias_start=None, bias_stop=None, bin_size=None, smooth=True, window=0.8) -> dict:
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
            'indices': indices,
            'bias_start': bias_start,
            'bias_stop': bias_stop,
            'bin_size': bin_size
        }

        spiketrains = [self.dataset.spiketrains[i] for i in range(len(self.dataset.spiketrains)) if i in indices]
        neuon_num = len(spiketrains)
        eventrain = eventrain[eventrain>=t_start]
        eventrain = eventrain[eventrain<t_stop]

        coarr = []
        coarridx = []
        for i in range(neuon_num):
            for j in range(i+1, neuon_num):
                target_bin = split([spiketrains[i]], eventrain, bias_start, bias_stop, bin_size).squeeze().T
                refer_bin = split([spiketrains[i]], eventrain, bias_start, bias_stop, bin_size).squeeze().T
                coarr_ij = co_matrix(target_bin, refer_bin)
                coarr.append(coarr_ij)
                coarridx.append([i, j])
        
        self.anares = {
            'coarr': np.stack(coarr),
            'coarridx': coarridx
        }

        return self.anares
    
    def plot(self, root, form='bar', **kwargs):
        """
        root: the saving root of result png files
        form: 'bar' or 'step' or 'curve'
        """
        n_units, t_steps = self.anares['coarr'].shape
        x = np.linspace(-self.params_data['bias_start'], self.params_data['bias_stop'], t_steps+1)
        x = x[:-1] + self.params_data['bin_size']/2
        
        fig = plt.figure()
        for j in range(n_units):
            axes = fig.add_subplot(1,1,1)
            sns.heatmap(ax=axes, data=self.anares["coarr"][j], **kwargs)
            idx = self.anares['coarridx'][j]
            file_name = str(idx[0]) + "_" + str(idx[1])
            fig.savefig(os.path.join(root, file_name+'.png'), transparent=True, dpi=200, pad_inches=0)
            plt.clf()
    
