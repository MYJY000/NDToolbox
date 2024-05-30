import numpy as np
from .basic import NeuralDataset
from .hdf_dao import HierarchicalFileLoader

class HDFNeuralDataset(NeuralDataset, HierarchicalFileLoader):
    """
    Nonhuman Primate Reaching with Multichannel Sensorimotor Cortex Electrophysiology
    Datasets Link: https://zenodo.org/records/3854034
    Example Data: https://zenodo.org/records/3854034/files/indy_20160426_01.mat?download=1
    """
    def __init__(self, path, **kwargs) -> None:
        HierarchicalFileLoader.__init__(self, path)
        NeuralDataset.__init__(self, **kwargs)

    def _load_behavior(self, behavior: str) -> np.ndarray:
        """load the raw behaviors data
        Returns
        ------
        target: 2D-ndarray(t, k)
        """
    
    def _load_behavior_binsize(self, **kwargs) -> float:
        """load the original smaple rates
        Returns
        ------
        old_target_binsize: float, e.g. 0.004 means for every 0.004s we sample a data point.
        """

    def load_behaviors(self, behavior, bin_size: float = None, t_start = None, t_stop = None) -> np.ndarray:
        """get behaviors data like knematic data `cursor_pos`
        `behavior`: str -- key
        `bin_size`: float -- resample knematic data to `bin_size`(seconds), e.g. 0.1
        [`t_start`, `t_stop`] -- only take the data between [`t_start`, `t_stop`]

        Returns
        ------
        `targets`: ndarray(t, k) -- behaviors data/knematic data
        """
        if t_start is None:
            t_start = self.t_start
        if t_stop is None:
            t_stop = self.t_stop
        t_start = max(t_start, self.t_start)
        t_stop = min(t_stop, self.t_stop)
        self.old_target_binsize = self._load_behavior_binsize()
        start_idx = int((t_start-self.t_start)/self.old_target_binsize)
        stop_idx = int((t_stop-self.t_start)/self.old_target_binsize)+1
        targets = self._load_behavior(behavior)
        targets = targets[start_idx: stop_idx]

        if bin_size is None:
            return targets
        else:
            self.current_target_binsize = bin_size
            assert int(bin_size*1000) % int(1000*self.old_target_binsize) == 0, \
            f"current sample binsize is {self.old_target_binsize}, new binsize is {bin_size}, please make sure {bin_size}/{self.old_target_binsize} be an integer"
            duration = (stop_idx-start_idx)*self.old_target_binsize
            t_steps = int(duration/bin_size)
            scale = int(bin_size/self.old_target_binsize)
            targets = targets[:t_steps*scale]
            targets = targets[::scale]
            return targets

            

