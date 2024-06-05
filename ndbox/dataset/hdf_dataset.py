import numpy as np
from .basic import NeuralDataset
from .hdf_dao import HierarchicalFileLoader
from .utils import idxs2train

class HDFNeuralDataset(NeuralDataset):
    def __init__(self, path, **kwargs) -> None:
        self.hdf_agent = HierarchicalFileLoader(path)
        NeuralDataset.__init__(self, **kwargs)
    
    def get_t_start(self, **kwargs) -> float:
        if self.t_start is not None:
            return self.t_start
        else:
            self.t_start = self.load('t_start')
            self.append('t_start', self.t_start)
            return self.load('t_start')
    
    def get_t_stop(self, **kwargs) -> float:
        if self.t_stop is not None:
            return self.t_stop
        else:
            self.t_stop = self.load('t_stop')
            self.append('t_stop', self.t_stop)
            return self.load('t_stop')
    
    def fetch_spiketrains(self, **kwargs) -> list[np.ndarray]:
        if self.spiketrains is not None:
            return self.spiketrains
        else:
            spiketime = self.load('spiketime')
            spikeindex = self.load('spikeindex')
            self.append('spiketime', spiketime)
            self.append('spikeindex', spikeindex)
            self.spiketrains = idxs2train(spiketime, spikeindex)
            return self.spiketrains
    
    def fetch_behaviors(self, key: str) -> np.ndarray:
        behavior = self.load(key)
        self.append(key, behavior)
        bin_size = self.load(key+"_bin_size")
        self.append(key+"_bin_size", bin_size)
        return behavior
    
    def fetch_event_stamps(self, key: str) -> np.ndarray:
        event = self.load(key)
        self.append(key, event)
        return event
    
    def keys(self) -> list[str]:
        return list(set(list(self.data_dict.keys()) + list(self.hdf_agent.keys())))
    
    def load(self, key: str) -> np.ndarray:
        val = None
        if key in self.data_dict:
            val = self.data_dict[key]
        elif key in list(self.hdf_agent.keys()):
            val = self.hdf_agent.load(key)
        if val is not None:
            if isinstance(val, list):
                if len(val) == 1:
                    return val[0]
            if isinstance(val, np.ndarray):
                if val.size == 1:
                    return val[0]
        return val       

