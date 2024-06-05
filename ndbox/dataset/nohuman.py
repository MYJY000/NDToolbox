import numpy as np
from .hdf_dataset import HDFNeuralDataset
from .utils import train2idxs


class NoHumanPR(HDFNeuralDataset):
    """
    Nonhuman Primate Reaching with Multichannel Sensorimotor Cortex Electrophysiology
    Datasets Link: https://zenodo.org/records/3854034
    Example Data: https://zenodo.org/records/3854034/files/indy_20160426_01.mat?download=1
    """
    def __init__(self, path, **kwargs) -> None:
        super(NoHumanPR, self).__init__(path, **kwargs)
    
    def get_t_start(self, **kwargs) -> float:
        if self.t_start is not None:
            return self.t_start
        else:
            self.t_start = round(self.load('t')[0, 0], 3)
            self.append('t_start', self.t_start)
            return self.load('t_start')
    
    def get_t_stop(self, **kwargs) -> float:
        if self.t_stop is not None:
            return self.t_stop
        else:
            self.t_stop = round(self.load('t')[0, -1], 3)
            self.append('t_stop', self.t_stop)
            return self.load('t_stop')
    
    def fetch_spiketrains(self, high_pass=2.0, **kwargs) -> list[np.ndarray]:
        """load spiketrains from given files.
        high_pass: if a unit's firing rates is higher than `high_pass`, then it is a valid unit.
        Example: 2.0 -- means that a valid unit's firing rates is at least 2.0 spike/seconds.
        
        Returns
        ------
        spiketrains: list[np.ndarray] -- A list of array, each array represents a unit's spikes
        """
        if self.spiketrains is not None:
            return self.spiketrains
        else:
            spikes = self.load('spikes')
            
            spiketrains = []
            duration = self.get_t_stop()-self.get_t_start()
            high_pass = int(duration * high_pass)
            r, c = spikes.shape
            for i in range(r):
                for j in range(c):
                    unit = spikes[i, j]
                    if unit.size > high_pass:
                        spiketrains.append(unit)
            
            spiketime, spikeindex = train2idxs(spiketrains)
            self.spiketrains = spiketrains
            self.append('spiketime', spiketime)
            self.append('spikeindex', spikeindex)
            return self.spiketrains
    
    def fetch_behaviors(self, key: str) -> np.ndarray:
        behavior = self.load(key).T
        self.append(key, behavior)
        bin_size = round(self.load('t')[0, 1] - self.load('t')[0, 0], 3)
        self.append(key+"_bin_size", bin_size)
        return behavior
