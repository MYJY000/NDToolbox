from abc import abstractmethod, ABCMeta
import numpy as np
from .utils import smooth2D, sample, split, smooth3D
import h5py

class NeuralDataset(metaclass=ABCMeta):
    def __init__(self, **kwargs) -> None:
        self.t_start = None
        self.t_stop = None
        self.spiketrains = None
        self.data_dict = {}     # 缓存数据字典
        self.get_t_start(**kwargs)
        self.get_t_stop(**kwargs)
        self.fetch_spiketrains(**kwargs)
        self.duration = self.t_stop - self.t_start
        self.append('duration', self.duration)
        self.neuron_counts = len(self.spiketrains)
        self.append('neuron_counts', self.neuron_counts)
        assert self.spiketrains is not None, "Spiketrains(List[np.ndarray]) is neccessary for initializing NeuralDataset object!"

    @abstractmethod
    def get_t_start(self, **kwargs) -> float:
        """load the begining timestamp of recording time or set values of `t_start`(set only when NeuralDataset is initialized)
        """

    @abstractmethod
    def get_t_stop(self, **kwargs) -> float:
        """load the ending timestamp of recording time or set values of `t_stop`(set only when NeuralDataset is initialized)
        """
        
    @abstractmethod
    def fetch_spiketrains(self, **kwargs) -> list[np.ndarray]:
        """load the spiketrains or set values of the `spiketrains`(set only when NeuralDataset is initialized)
        """
    
    def fetch_behaviors(self, **kwargs) -> np.ndarray:
        """Load the behaviors data or set values of the behaviors data.
        Example: `cursor_pos`, `hand_pos`, `hand_vel`, ... etc.
        Return: 
            behaviors: (t, k) -- the behavior data for each time step
        """
    
    def fetch_event_stamps(self, **kwargs) -> np.ndarray:
        """Load the event timestamps or set values of the behaviors data.
        Example: `move_onset_time`, `go_cue`, `start_time`, `end_time`... etc.
        Return: (t,) -- represents the time stamps of each event
        """
    
    def resample_behaviors(self, key, bin_size, t_start=None, t_stop=None) -> np.ndarray:
        """Resample the behavior data to `bin_size`
        """
        if t_start is None:
            t_start = self.t_start
        if t_stop is None:
            t_stop = self.t_stop
        t_start = max(t_start, self.t_start)
        t_stop = min(t_stop, self.t_stop)

        behavior = self.load(key)
        bin_size_old = self.load(key+"_bin_size")

        start_idx = int((t_start-self.t_start)/bin_size_old)
        stop_idx = int((t_stop-self.t_start)/bin_size_old)+1
        targets = behavior[start_idx: stop_idx]

        assert int(bin_size*1000) % int(1000*bin_size_old) == 0, f"original binsize is {bin_size_old}, new binsize is {bin_size}, which is not divisible."
        duration = (stop_idx-start_idx)*bin_size_old
        t_steps = int(duration/bin_size)
        scale = int(bin_size/bin_size_old)
        targets = targets[:t_steps*scale]
        targets = targets[::scale]
        return targets

    def sample_spiketrains(self, bin_size, t_start=None, t_stop=None, smooth=True, window=0.8) -> np.ndarray:
        """Sample the spiketrains to `bin_size` from `t_start`(unit: seconds) to `t_stop`(unit: seconds)

        Notes
        ------
        `bin_size` is given by behaviors data's sampling rates
        
        Returns
        ------
        `firing_rates`: (t, n) -- binned spike array(or firing rates if `smooth` is `Ture`). `t` is time steps, `n` is unit(neuron/channel) counts
        """
        if t_start is None:
            t_start = self.t_start
        if t_stop is None:
            t_stop = self.t_stop
        spike_array = sample(self.spiketrains, bin_size, t_start, t_stop)
        if smooth:
            spike_array = smooth2D(spike_array, bin_size, window)
        self.spike_binsize = bin_size
        self.firing_rates = spike_array
        return spike_array

    def split(self, stimulus, bias_start, bias_stop, bin_size, smooth=True, window=0.8) -> np.ndarray:
        """Split each spiketrain in spiketrains using given event(stimulus/mark/flag) time series.
        
        Returns
        ------
        `spikepochs`: np.ndarray(n, t, s)
            `spikepochs[i, j, k]` represents for neuron/unit/channel `i`, the firing rates in the `j`th bin around the event`k`.
        """
        spike_array = split(self.spiketrains, stimulus, bias_start, bias_stop, bin_size)
        if smooth:
            return smooth3D(spike_array, bin_size, window)
        else:
            return spike_array
    
    def keys(self) -> list[str]:
        """Return the data_dict keys that have been loaded
        """
        return list(self.data_dict.keys())
    
    def append(self, path: str, value: np.ndarray[int] | np.ndarray[float] | str | int | float | list[int] | list[float] | list[str]):
        """Append a new item to data_dict
        """
        if isinstance(value, (str, int, float)):
            value = [value]
        assert isinstance(value, (np.ndarray, list))
        if path not in self.data_dict:
            self.data_dict[path] = value

    def load(self, key: str) -> np.ndarray:
        """Return the item in data_dict with the corresponding `key`
        """
        val = self.data_dict[key]
        if isinstance(val, list):
            if len(val) == 1:
                return val[0]
        if isinstance(val, np.ndarray):
            if val.size == 1:
                return val[0]
        return val
    
    def save(self, path: str):
        """Save the data_dict(only the data you've loaded and appended)
        """
        file = h5py.File(path, 'w')
        for k in self.data_dict:
            file.create_dataset(k, data=self.data_dict[k])
        file.close()
