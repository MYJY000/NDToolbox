import numpy as np

def smooth1D(spike_array: np.ndarray, bin_size: float, window: float):
    from scipy.signal import windows
    window_size = int(window/bin_size)
    window = windows.gaussian(window_size, window_size, sym=True)
    window /= np.sum(window)
    return np.convolve(spike_array, window, 'same')


def smooth2D(spike_array: np.ndarray, bin_size: float, window: float):
    """Smooth the spike counts with gusssian covolution.

    Parameters
    ------
    `spike_array`: ndarray(t, n)
        binned spike array(e.g. return value of function `sample`).
    `bin_size`: float
        Sample binsize of `spike_array`, e.g. `0.1`(seconds).
    `window`: float
        size of smooth window, e.g. `0.8`(seconds)(so here we smooth with 8 bins). There larger the `window/bin_size`, the flatter the firing rates.
    
    Returns
    ------
    `firing_rates`: ndarray(t, n)
        smoothed binned spike array.
    """
    spike_array = spike_array.astype(np.float32)
    smooth_func = lambda x: smooth1D(x, bin_size, window)
    return np.apply_along_axis(smooth_func, 0, spike_array)

def smooth3D(spike_array: np.ndarray, bin_size: float, window: float):
    """Smooth the spike counts with gusssian covolution.

    Parameters
    ------
    `spike_array`: np.ndarray(n, t, s)
        binned spike array(e.g. return value of function `sample`).
    `bin_size`: float
        Sample binsize of `spike_array`, e.g. `0.1`(seconds).
    `window`: float
        size of smooth window, e.g. `0.8`(seconds)(so here we smooth with 8 bins). There larger the `window/bin_size`, the flatter the firing rates.
    `t_stop`: float
        The finishing timestamp.
    
    Returns
    ------
    `firing_rates`: ndarray(n, t, s)
        smoothed binned spike array.
    """
    spike_array = spike_array.astype(np.float32)
    for i in range(spike_array.shape[0]):
        spike_array[i] = smooth2D(spike_array[i], bin_size, window)
    return spike_array

def sample(spiketrains, bin_size, t_start, t_stop):
    """Sample the spiketrains to `bin_size`(unit: seconds) from `t_start`(unit: seconds) to `t_stop`(unit: seconds)

    Parameters
    ------
    `spiketrains`: list[np.ndarray](n,)
        A list of ndarray, each elements(like `spiketrains[i]`) represents spike timestamps of one unit/neuron/channel
    `bin_size`: float
        Sample binsize of `spike_array`, e.g. `0.1`(seconds)
    `t_start`: float
        The beginning timestamp.
    `t_stop`: float
        The finishing timestamp.
    
    Returns
    ------
    `spike_array`: ndarray(t, n)
        binned spike array.
    """
    spike_array = []
    t_steps = int(((t_stop-t_start)*1000)/(1000*bin_size))
    for sp in spiketrains:
        hist, edge = np.histogram(sp, t_steps, (t_start, t_start+t_steps*bin_size))
        spike_array.append(hist)
    spike_array = np.stack(spike_array).T
    return spike_array

def split(spiketrains, stimulus, bias_start, bias_stop, bin_size, decentralization=False):
    """Split each spiketrain in spiketrains using given event(stimulus/mark/flag) time series.

    Parameters
    ------
    `spiketrains`: list[np.ndarray](n,)
        A list of ndarray, each elements(like `spiketrains[i]`) represents spike timestamps of one unit/neuron/channel
    `stimulus`: np.ndarray(s,)
        The stimulus time sequence.
    `bias_start`: float
        The time bias before the event.
    `bias_stop`: float
        The time bias after the event.
    `bin_size`: float
        Sample binsize of `spike_array`, e.g. `0.1`(seconds).
    `decentralization`: bool
        Sometimes the `stimulus` may coincides with part of `spiketrains`, so need decentralization.

    Returns
    ------
    `spikepochs`: np.ndarray(n, t, s)
        `spikepochs[i, j, k]` represents for neuron/unit/channel `i`, whether it fires in the `j`th bin around the event`k`.
    """
    t_start = 0
    t_stop = bias_start+bias_stop
    spikepochs = []
    for sp in spiketrains:
        sp_trains = []
        for sti in stimulus:
            sp_around = sp-(sti-bias_start)
            sp_around = sp_around[sp_around>=t_start]
            sp_around = sp_around[sp_around<t_stop]
            if decentralization:
                sp_around = sp_around[np.abs(sp_around-bias_start)>1e-4]
            sp_trains.append(sp_around)
        spikepochs.append(sample(sp_trains, bin_size, t_start, t_stop))
    return np.stack(spikepochs)

def cut(spiketrains, t_start, t_stop) -> list[np.ndarray]:
    """Cut each spiketrain in spiketrains so that the firing timestamp between [t_start, t_stop)
    """
    spt = []
    for sp in spiketrains:
        s = sp[sp>=t_start]
        s = s[s<t_stop]
        spt.append(s)
    return spt

def bypass(spiketrains, duration, high_pass, low_pass) -> list[int]:
    """Select the spiketrain idx whose firing rates is in [high_pass, low_pass]
    """
    spidx = []
    high_pass = int(duration * high_pass)
    low_pass = int(duration * low_pass)
    for i in range(len(spiketrains)):
        sp = spiketrains[i]
        if sp.size >= high_pass and sp.size <= low_pass:
            spidx.append(i)
    return spidx

def train2idxs(spiketrains) -> tuple[np.ndarray, np.ndarray]:
    """cast `spiketrains` to `spiketime` and `spikeindex`
    """
    size_list = [st.size for st in spiketrains]
    spikeindex = np.cumsum(np.array(size_list))
    spiketime = np.hstack(spiketrains).squeeze()
    return spiketime, spikeindex

def idxs2train(spiketime, spikeindex) -> list[np.ndarray]:
    """cast `spiketime` and `spikeindex` to `spiketrains`
    """
    prev = 0
    spiketrains = []
    for post in spikeindex:
        spiketrains.append(spiketime[prev: post])
        prev = post
    return spiketrains

