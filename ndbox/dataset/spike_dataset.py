import numpy as np
from .basic import NeuralDataset

class SpikeTrainDataset(NeuralDataset):
    def __init__(self, spiketrains: list[np.ndarray], t_start=None, t_stop=None) -> None:
        super(SpikeTrainDataset, self).__init__(spiketrains=spiketrains, t_start=t_start, t_stop=t_stop)
        self.spiketrains = spiketrains
    
    def get_t_start(self, t_start: float = 0, **kwargs) -> float:
        if t_start is not None:
            self.t_start = t_start
            return t_start
        else:
            pass
    
    def get_t_stop(self, t_stop: float = 0, **kwargs) -> float:
        if t_stop is not None:
            self.t_stop = t_stop
            return t_stop
        else:
            pass
    
    def load_behaviors(self, behaviors: np.ndarray, **kwargs) -> np.ndarray:
        self.behaviors = behaviors
        return behaviors
    
    def load_event_stamps(self, event_stamps: np.ndarray, **kwargs) -> np.ndarray:
        self.event_stamps = event_stamps
        return event_stamps
    
    def load_spiketrains(self, spiketrains: list[np.ndarray], **kwargs) -> list[np.ndarray]:
        self.spiketrains = spiketrains
        return spiketrains
    