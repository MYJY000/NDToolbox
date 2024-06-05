import numpy as np
from scipy.optimize import curve_fit
import numpy as np

def pos2radian(position):
    """Transfer `position` to `radian`[0, 2*pi]"""
    pos_x = position[:, 0]
    pos_y = position[:, 1]
    pox_diff = np.diff(pos_x)
    poy_diff = np.diff(pos_y)
    theta = np.arctan(poy_diff / pox_diff)
    theta[pox_diff < 0] += np.pi
    theta[theta < 0] += 2 * np.pi
    return theta

def cos_tuning_model(x, m, pd, b0):
    return m * np.cos(x - pd) + b0

def cos_fit(x, y):
    fmn = y.min()
    fmx = y.max()
    # noinspection PyTupleAssignmentBalance
    params, _ = curve_fit(cos_tuning_model, x, y,
                          p0=[(fmn - fmx) / 2, 1, fmn + 1],
                          bounds=([-np.inf, 0, -np.inf], [np.inf, 2 * np.pi, np.inf]))
    return params

def prefer_direction(m, pd, b0):
    if m > 0:
        return pd
    else:
        return pd + np.pi

class CosineTuningModel:
    def __init__(self, n_direction=8) -> None:
        """
        n_direction: int
            The number of data points used to fit cosine curve
        """
        self.n_direction = n_direction
        delta = 2 * np.pi / self.n_direction
        self.angle = np.zeros(self.n_direction + 1)
        for i in range(self.n_direction):
            self.angle[i + 1] = delta * (i + 1)
    
    def fit(self, firing_rates, position, bin_size):
        """
        firing_rates: (t, )
        position: (t, 2)
        """
        direction = pos2radian(position)
        data_length = min(direction.size, firing_rates.size)
        direction = direction[:data_length]
        firing_rates = firing_rates[:data_length]
        fl = ~np.isnan(direction)
        direction = direction[fl]
        firing_rates = firing_rates[fl]
        
        x = []
        y = []
        duration = direction.shape[0] * bin_size
        for i in range(self.n_direction):
            epoch_mask = (direction >= self.angle[i]) & (direction < self.angle[i + 1])
            theta_epoch = direction[epoch_mask]
            fr_epoch = firing_rates[epoch_mask]
            x.append(theta_epoch.mean())
            y.append(fr_epoch.sum()/duration)

        self.tuning_x = np.array(x)
        self.tuning_y = np.array(y)
        self.params = cos_fit(self.tuning_x, self.tuning_y)
        pd = prefer_direction(*self.params)
        return pd
