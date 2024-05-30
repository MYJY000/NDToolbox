from abc import abstractmethod, ABCMeta
import numpy as np
from .utils import smooth2D, sample, split, smooth3D

class NeuralDataset(metaclass=ABCMeta):
    def __init__(self, **kwargs) -> None:
        self.t_start = self.get_t_start(**kwargs)
        self.t_stop = self.get_t_stop(**kwargs)
        self.spiketrains = self.load_spiketrains(**kwargs)
        self.duration = self.t_stop - self.t_start
        self.unit_counts = len(self.spiketrains)
        assert self.spiketrains is not None, "Spiketrains(List[np.ndarray]) is neccessary for initializing NeuralDataset object!"

    @abstractmethod
    def get_t_start(self, t_start: float = 0, **kwargs) -> float:
        """load the start timestamp of recording time or set it to `default`
        """

    @abstractmethod
    def get_t_stop(self, t_stop: float = 0, **kwargs) -> float:
        """load the start timestamp of recording time or set it to `default`
        """
    
    @abstractmethod
    def load_spiketrains(self, **kwargs) -> list[np.ndarray]:
        """load the spiketrains or set values of the spiketrains
        """
    
    
    def load_behaviors(self, **kwargs) -> np.ndarray:
        """Optional, load the behaviors data or set values of the behaviors data.
        Example: `cursor_pos`, `hand_pos`, `hand_vel`, ... etc.
        """
    
    def load_event_stamps(self, **kwargs) -> np.ndarray:
        """Optional, load the behaviors data or set values of the behaviors data.
        Example: `move_onset_time`, `go_cue`, `start_time`, `end_time`... etc.
        Return: (t,) -- represents the time stamps of each event
        or (t, 2) -- represents the [start, stop] of each event
        """

    def sample(self, bin_size, t_start=None, t_stop=None, smooth=True, window=0.8):
        """Sample the spiketrains to `bin_size`(unit: seconds) from `t_start`(unit: seconds) to `t_stop`(unit: seconds)

        Example
        ------
        >>> data.sample(0.1)
        
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

    def split(self, stimulus, bias_start, bias_stop, bin_size, smooth=True, window=0.8):
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
