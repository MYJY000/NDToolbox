import numpy as np
from .hdf_dataset import HDFNeuralDataset
import warnings


class DANDI(HDFNeuralDataset):
    """
    Nonhuman Primate Reaching with Multichannel Sensorimotor Cortex Electrophysiology
    Archives: https://dandiarchive.org/dandiset
    Example Data:
    https://dandiarchive.org/dandiset/000129/draft
    or
    https://api.dandiarchive.org/api/assets/2ae6bf3c-788b-4ece-8c01-4b4a5680b25b/download/
    """
    def __init__(self, path, **kwargs) -> None:
        super(DANDI, self).__init__(path, **kwargs)
    
    def get_t_start(self, t_start: float = 0, **kwargs) -> float:
        try:
            start_time = self.load('intervals/trials/start_time')
            return start_time[0]
        except:
            try:
                return self.t_start
            except:
                return np.nan
    
    def get_t_stop(self, t_stop: float = 0, **kwargs) -> float:
        try:
            stop_time = self.load('intervals/trials/stop_time')
            return stop_time[-1]
        except:
            try:
                return self.t_stop
            except:
                return np.nan
    
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
        try:
            spike_times = self.load('units/spike_times')
            spike_times_index = self.load('units/spike_times_index')
            spiketrains = np.array_split(spike_times, spike_times_index)
            if np.isnan(self.t_start):
                self.t_start = spike_times[0]
            if np.isnan(self.t_stop):
                self.t_stop = spike_times[-1]
            self.t_start = round(self.t_start, 3)
            self.t_stop = round(self.t_stop, 3)
            self.duration = self.t_stop - self.t_start
            high_pass = int(self.duration * high_pass)
            return [sp for sp in spiketrains if sp.size > high_pass]
        except Exception as e:
            print(e)
            raise ValueError("Group `units` not found! You may check if there contains spikes in the file.")
    
    def load_event_stamps(self, event: str) -> np.ndarray:
        """load event time stamps from the dataset.

        envents: ndarray(k,) -- contains `k` event timestamps
        """
        return self.load(event)

    def _load_behavior_binsize(self, **kwargs) -> float:
        return 0.001
    
    def _load_behavior(self, behavior: str) -> np.ndarray:
        return self.load(behavior)
