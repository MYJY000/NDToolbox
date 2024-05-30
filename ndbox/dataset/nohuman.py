import numpy as np
from .hdf_dataset import HDFNeuralDataset

class NoHumanPR(HDFNeuralDataset):
    """
    Nonhuman Primate Reaching with Multichannel Sensorimotor Cortex Electrophysiology
    Datasets Link: https://zenodo.org/records/3854034
    Example Data: https://zenodo.org/records/3854034/files/indy_20160426_01.mat?download=1
    """
    def __init__(self, path, **kwargs) -> None:
        super(NoHumanPR, self).__init__(path, **kwargs)

    def load_spiketrains(self, high_pass=None) -> list[np.ndarray]:
        """load spiketrains from given files.
        high_pass: if a unit's firing rates is higher than `high_pass`, then it is a valid unit.
        Example: 2.5 -- means that a valid unit's firing rates is at least 2.5 spike/seconds.
        
        Returns
        ------
        spiketrains: list[np.ndarray] -- A list of array, each array represents a unit's spikes
        """
        if high_pass is None:
            high_pass = 2
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
        return spiketrains
    
    def get_t_start(self, t_start: float = 0, **kwargs) -> float:
        return round(self.load('t')[0, 0], 3)
    
    def get_t_stop(self, t_stop: float = 0, **kwargs) -> float:
        return round(self.load('t')[0, -1], 3)
    
    def _load_behavior(self, behavior: str) -> np.ndarray:
        return self.load(behavior).T
    
    def _load_behavior_binsize(self, **kwargs) -> float:
        return round(self.load('t')[0, 1] - self.load('t')[0, 0], 3)
