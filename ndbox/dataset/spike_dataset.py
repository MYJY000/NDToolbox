import numpy as np
from .basic import NeuralDataset
from .utils import train2idxs


class SpikeTrainDataset(NeuralDataset):
    def __init__(self, spiketrains: list[np.ndarray], t_start, t_stop) -> None:
        """To initialize a SpikeTrainDataset, you should give a list of timeseries(`spiketrains`)
        which represents a number of neuron's firing timestamps and start as well as stop time of 
        `spiketrains`.

        Attetion
        ------
        Make sure the unit of firing timestamps be 's'(seconds)
        """
        super(SpikeTrainDataset, self).__init__(spiketrains=spiketrains, t_start=t_start, t_stop=t_stop)
        self.spiketrains = spiketrains
    
    def get_t_start(self, t_start: float = None, **kwargs) -> float:
        if self.t_start is not None:
            return self.t_start
        if t_start is not None:
            self.t_start = t_start
            self.append('t_start', self.t_start)
            return t_start
    
    def get_t_stop(self, t_stop: float = None, **kwargs) -> float:
        if self.t_stop is not None:
            return self.t_stop
        if t_stop is not None:
            self.t_stop = t_stop
            self.append('t_stop', self.t_stop)
            return t_stop
    
    def fetch_spiketrains(self, spiketrains: list[np.ndarray] = None, **kwargs) -> list[np.ndarray]:
        if self.spiketrains is not None:
            return self.spiketrains
        if spiketrains is not None:
            self.spiketrains = spiketrains
            spiketime, spikeindex = train2idxs(self.spiketrains)
            self.append('spiketime', spiketime)
            self.append('spikeindex', spikeindex)
            return spiketrains
    
    def fetch_behaviors(self, key: str, behaviors: np.ndarray = None, bin_size: float = None) -> np.ndarray:
        """Load the behaviors data or set values of the behaviors data.

        Params:
        ------
            behaviors: (t, k) -- the behavior data for each time step
            bin_size: float -- the behaviors data sampling rates
            key: e.g. `cursor_pos`, `hand_pos`, `hand_vel`, ... etc.
        
        Notes:
        ------
            If `key` is already in data_dict, then the behaviors data will be loaded and return.
            If not, then behaviors data will store in `data_dict` with a key `key`.
        
        Warning:
        ------
            You should make sure `t*bin_size == (t_start-t_stop)` almostly.
        """
        if key in self.data_dict:
            return self.load(key)
        elif behaviors is not None and bin_size is not None:
            t = behaviors.shape[0]
            assert behaviors.ndim == 2, "behaviors: (t, k) -- the behavior data for each time step"
            self.append(key, behaviors)
            self.append(key+'_bin_size', bin_size)
            return self.load(key)
    
    def fetch_event_stamps(self, key: str, event: np.ndarray = None) -> np.ndarray:
        if key in self.data_dict:
            return self.load(key)
        elif event is not None:
            self.append(key, event)
            return self.load(key)
    
    