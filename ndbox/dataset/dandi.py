import numpy as np
from .hdf_dataset import HDFNeuralDataset
from .utils import idxs2train, train2idxs


class DANDI(HDFNeuralDataset):
    """
    DANDI is an archive containning many nwb datasets
    Archives: https://dandiarchive.org/dandiset
    Example Data:
    https://dandiarchive.org/dandiset/000129/draft
    or
    https://api.dandiarchive.org/api/assets/2ae6bf3c-788b-4ece-8c01-4b4a5680b25b/download/
    """
    def __init__(self, path, **kwargs) -> None:
        super(DANDI, self).__init__(path, **kwargs)
    
    def get_t_start(self, **kwargs) -> float:
        if self.t_start is not None:
            return self.t_start
        else:
            try:
                start_time = self.load('intervals/trials/start_time')[0]
            except:
                start_time = np.nan
            self.t_start = start_time
            self.append('t_start', self.t_start)
            return self.load('t_start')
    
    def get_t_stop(self, **kwargs) -> float:
        if self.t_stop is not None:
            return self.t_stop
        else:
            try:
                stop_time = self.load('intervals/trials/stop_time')[-1]
            except:
                stop_time = np.nan
            self.t_stop = stop_time
            self.append('t_stop', self.t_stop)
            return self.load('t_stop')
    
    def fetch_spiketrains(self, high_pass=2.0) -> list[np.ndarray]:
        """load spiketrains from given files.
        `high_pass`: if a unit's firing rates is higher than `high_pass`, then it is a valid unit.
        Example: 2.0 -- means that a valid unit's firing rates is at least 2.0 spike/seconds.
        Notes: `high_pass` only effect when initialize the dataset.
        
        Returns
        ------
        spiketrains: list[np.ndarray] -- A list of array, each array represents a unit's spikes
        """
        if self.spiketrains is not None:
            return self.spiketrains
        try:
            spiketime  = self.load('units/spike_times')
            spikeindex = self.load('units/spike_times_index')

            spiketrains = idxs2train(spiketime, spikeindex)
            if np.isnan(self.t_start):
                self.t_start = round(spiketime[0], 3)
                self.data_dict['t_start'] = self.t_start
            if np.isnan(self.t_stop):
                self.t_stop = round(spiketime[-1], 3)
                self.data_dict['t_stop'] = self.t_stop
            high_pass = int((self.t_stop-self.t_start) * high_pass)
            spiketrains = [sp for sp in spiketrains if sp.size > high_pass]

            spiketime, spikeindex = train2idxs(spiketrains)
            self.spiketrains = spiketrains
            self.append('spiketime', spiketime)
            self.append('spikeindex', spikeindex)
            return self.spiketrains
        except Exception as e:
            print(e)
            raise ValueError("You may check if there contains spikes in the file. Or you can follow the tips to write your own NeuralDatasets.")

    def fetch_behaviors(self, key: str) -> np.ndarray:
        behavior = self.load(key)
        self.append(key, behavior)
        bin_size = 0.001
        self.append(key+"_bin_size", bin_size)
        return behavior

