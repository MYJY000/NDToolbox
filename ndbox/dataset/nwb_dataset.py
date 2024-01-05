import os
import numpy as np
import pandas as pd
from copy import deepcopy
from collections import OrderedDict
from pynwb import NWBHDF5IO, TimeSeries
from pynwb.core import MultiContainerInterface, DynamicTable

from ndbox.utils import get_root_logger, DATASET_REGISTRY, dict2yaml, yaml2dict


@DATASET_REGISTRY.register()
class NWBDataset:
    """
    A class for loading datasets from NWB files.
    """

    def __init__(self, path, name='NWBDataset', units=None, trials=None, behavior=None,
                 spike_identifier='spike#', skip_fields=None, image_path=None, **kwargs):
        """
        Initializes an NWBDataset, loading datasets from
        the indicated file(s).
        """

        self.name = name
        self.logger = get_root_logger()
        self.logger.info(f"Loading dataset '{name}' from file '{path}'")
        if not os.path.exists(path):
            raise FileNotFoundError(f"'{path}' File not found!")
        if os.path.isdir(path):
            raise FileNotFoundError(f"The specified path '{path}' is a "
                                    f"directory， requires a file")
        self.filename = path

        self.units_opt = units if units is not None else {}
        self.trials_opt = trials if trials is not None else {}
        self.behavior_list = behavior if behavior is not None else []
        self.spike_identifier = spike_identifier
        self.smooth_identifier = 'smooth#'
        self.split_identifier = 'split#'
        self.skip_fields = skip_fields if skip_fields is not None else []

        self.data_dict, self.content_dict = self._build_data()

        self.spike_train = None
        self.bin_size = 0.001
        self.data = None
        self.trials = None
        self.behavior_columns = {}
        self.image = {}

        if image_path is not None:
            self.restore_image(image_path)
        else:
            self.logger.info(f"{self.content_repr(self.content_dict)}")
            self.logger.info(f"{self.data_info()}")
            self._load_data()

    def _build_data(self):
        io = NWBHDF5IO(self.filename, 'r')
        nwb_file = io.read()
        data_dict, content_dict = self._find_items(nwb_file, '')
        return data_dict, content_dict

    def _find_items(self, obj, prefix):
        data_dict = {}
        content_dict = {}
        for child in obj.children:
            name = prefix + child.name
            if name in self.skip_fields:
                continue
            content_dict[child.name] = {}
            if isinstance(child, MultiContainerInterface):
                d1, d2 = self._find_items(child, name + '/')
                data_dict.update(d1)
                content_dict[child.name] = d2
            else:
                data_dict[name] = child
        return data_dict, content_dict

    def content_repr(self, data, rank=0):
        msg = '\n'
        if not isinstance(data, dict):
            return ''
        for key, value in data.items():
            if not isinstance(value, dict) or len(value) == 0:
                msg += '|   ' * rank + str(key) + '\n'
            else:
                msg += '|   ' * rank + str(key) + '/'
                msg += self.content_repr(value, rank + 1)
        return msg

    def data_info(self):
        msg = '\n'
        for val in self.data_dict.values():
            msg += f'{val}\n'
        return msg

    def make_data(self, field):
        obj = self.data_dict.get(field)
        if obj is None:
            raise KeyError(f"No field '{field}' in datasets dict!")
        elif isinstance(obj, TimeSeries):
            if obj.timestamps is not None:
                time_index = np.array(obj.timestamps[()])
            else:
                time_index = np.arange(obj.data.shape[0]) / obj.rate + obj.starting_time
            index = time_index.round(6)
            columns = []
            if len(obj.data.shape) == 2:
                for i in range(obj.data.shape[1]):
                    columns.append(obj.name + '_' + str(i))
            elif len(obj.data.shape) == 1:
                columns.append(obj.name)
            else:
                self.logger.warning(f"Not support datasets dims larger than 2, "
                                    f"'{obj.name}' shape is '{obj.data.shape}'")
                return obj.data[()]
            df = pd.DataFrame(obj.data[()], index=index, columns=columns)
            return df
        elif isinstance(obj, DynamicTable):
            df = obj.to_dataframe()
            return df
        self.logger.warning(f"The '{obj.name}' class '{obj.__class__.__name__}' not support!")
        return None

    def _load_data(self):
        start_time = 0
        stop_time = 0

        self.logger.info(f"Load behavior datasets.")
        behavior_data_dict = {}
        for behavior_field in self.behavior_list:
            behavior_data = self.make_data(behavior_field)
            if len(behavior_data.shape) > 2:
                continue
            end_time = round(float(behavior_data.index[-1]), 6)
            stop_time = max(stop_time, end_time)
            behavior_data.index = pd.to_timedelta(behavior_data.index, unit='s')
            behavior_data_dict[behavior_field] = behavior_data

        self.logger.info(f"Load trials datasets.")
        trials_field = self.trials_opt.get('field', 'trials')
        trial_start = self.trials_opt.get('start', 'start_time')
        trial_stop = self.trials_opt.get('stop', 'stop_time')
        trials_obj = self.data_dict.get(trials_field)
        if trials_obj is None:
            self.logger.warning("Trials datasets not found in dataset!")
        else:
            trials = trials_obj.to_dataframe()
            end_time = trials[trial_stop].iloc[-1]
            stop_time = max(stop_time, end_time)
            trials[trial_start] = pd.to_timedelta(trials[trial_start], unit='s')
            trials[trial_stop] = pd.to_timedelta(trials[trial_stop], unit='s')
            self.trials = trials

        self.logger.info(f"Load units datasets.")
        units_field = self.units_opt.get('field')
        bin_size = self.units_opt.get('bin_size', 0.001)
        if units_field is None:
            if 'Units' in self.content_dict:
                units_field = 'Units'
            elif 'units' in self.content_dict:
                units_field = 'units'
            else:
                self.logger.warning("Units datasets not found in dataset!")
                return
        units_obj = self.data_dict.get(units_field)
        if isinstance(units_obj, DynamicTable):
            units = units_obj.to_dataframe()
            if 'spike_times' not in units.columns:
                self.logger.warning("Spike times datasets not found in units!")
                return
            self.spike_train = deepcopy(np.array(units['spike_times']))
            if 'obs_intervals' in units.columns:
                end_time = max(units.obs_intervals.apply(lambda x: x[-1][-1]))
                stop_time = max(stop_time, end_time)
            for _, unit in units.iterrows():
                end_time = max(unit.spike_times)
                stop_time = max(stop_time, end_time)
            timestamps = (np.arange(start_time, stop_time, round(bin_size, 6))).round(6)
            timestamps_td = pd.to_timedelta(timestamps, unit='s')
            spike_counts = np.full((len(timestamps), len(units)), 0.0, dtype='float64')
            for idx, (_, unit) in enumerate(units.iterrows()):
                hist, bins = np.histogram(unit.spike_times, bins=len(timestamps),
                                          range=(timestamps[0], timestamps[-1]))
                spike_counts[:, idx] = hist
            columns = units.index
            max_length = 1
            for c in columns:
                max_length = max(max_length, len(str(c)))
            columns = [self.spike_identifier + str(c).zfill(max_length) for c in columns]
            units_data = pd.DataFrame(spike_counts, index=timestamps_td,
                                      columns=columns).astype('float64', copy=False)
        elif isinstance(units_obj, TimeSeries):
            if units_obj.rate is None:
                self.logger.warning(f"The units is a Timeseries class, "
                                    f"the rate attribute must be included.")
                return
            bin_size = round(1. / units_obj.rate, 3)
            self.logger.info(f"Units datasets contains rate {units_obj.rate} Hz, "
                             f"bin_size is set to {bin_size} seconds.")
            stop_time = round(units_obj.data.shape[0] * bin_size, 6)
            spike_counts = deepcopy(np.array(units_obj.data[()]))
            if str(units_obj.unit) == 'Hz':
                spike_counts = (spike_counts * bin_size).round(0)
            timestamps = (np.arange(start_time, stop_time, round(bin_size, 6))).round(6)
            timestamps_td = pd.to_timedelta(timestamps, unit='s')
            max_length = len(str(spike_counts.shape[1]))
            columns = [self.spike_identifier + str(c).zfill(max_length)
                       for c in range(spike_counts.shape[1])]
            units_data = pd.DataFrame(spike_counts, index=timestamps_td,
                                      columns=columns).astype('float64', copy=False)
        else:
            self.logger.warning(f"Units datasets class not support!")
            return

        self.logger.info(f"Alignment timestamp.")
        data = units_data
        for key, value in behavior_data_dict.items():
            self.logger.info(f"Alignment '{key}' datasets timestamp.")
            data = pd.merge_asof(data, value, left_index=True, right_index=True)
            self.behavior_columns[key] = value.columns
        data.index.name = 'clock_time'
        data.sort_index(axis=1, inplace=True)
        self.bin_size = bin_size
        self.data = data
        self.logger.info(f"Load datasets successful.")

    def get_spike_array(self):
        smooth_mask = self.data.columns.str.startswith(self.smooth_identifier)
        smooth_columns = self.data.columns[smooth_mask]
        if len(smooth_columns) > 0:
            data = self.data[smooth_columns].to_numpy()
            return smooth_columns, data
        else:
            spike_mask = self.data.columns.str.startswith(self.spike_identifier)
            spike_columns = self.data.columns[spike_mask]
            data = self.data[spike_columns].to_numpy()
            return spike_columns, data

    def get_behavior_array(self, behavior_list):
        columns = []
        for item in behavior_list:
            columns.extend(self.behavior_columns[item])
        data = self.data[columns].to_numpy()
        return columns, data

    def get_spike_and_other_columns(self):
        spike_mask = self.data.columns.str.startswith(self.spike_identifier)
        smooth_mask = self.data.columns.str.startswith(self.smooth_identifier)
        spike_columns = self.data.columns[spike_mask | smooth_mask]
        other_columns = self.data.columns[~(spike_mask | smooth_mask)]
        return spike_columns, other_columns

    def get_data_startswith(self, identifier):
        mask = self.data.columns.str.startswith(identifier)
        columns = self.data.columns[mask]
        data = self.data[columns].to_numpy()
        return columns, data

    def drop_smooth_columns(self):
        drop_mask = self.data.columns.str.startswith(self.smooth_identifier)
        drop_columns = self.data.columns[drop_mask]
        if len(drop_columns) > 0:
            self.data.drop(drop_columns, axis=1, inplace=True)

    def save_image(self, path=None):
        if path is None:
            self.image['spike_train'] = deepcopy(self.spike_train)
            self.image['bin_size'] = deepcopy(self.bin_size)
            self.image['data'] = deepcopy(self.data)
            self.image['trials'] = deepcopy(self.trials)
            self.image['behavior_columns'] = deepcopy(self.behavior_columns)
        else:
            spike_train_path = os.path.join(path, 'spike_train.npy')
            bin_size_path = os.path.join(path, 'bin_size.txt')
            data_path = os.path.join(path, 'data.csv')
            trials_path = os.path.join(path, 'trials.csv')
            behavior_columns_path = os.path.join(path, 'behavior_columns.yml')
            np.save(spike_train_path, self.spike_train)
            with open(bin_size_path, 'w', encoding='utf-8') as f:
                f.write(str(self.bin_size))
            self.data.to_csv(data_path)
            self.trials.to_csv(trials_path)
            with open(behavior_columns_path, 'w', encoding='utf-8') as f:
                f.write(dict2yaml(OrderedDict(self.behavior_columns)))
            self.logger.info(f"Saving image to {path}")

    def restore_image(self, path=None):
        if path is None:
            self.spike_train = self.image['spike_train']
            self.bin_size = self.image['bin_size']
            self.data = self.image['data']
            self.trials = self.image['trials']
            self.behavior_columns = self.image['behavior_columns']
        else:
            spike_train_path = os.path.join(path, 'spike_train.npy')
            bin_size_path = os.path.join(path, 'bin_size.txt')
            data_path = os.path.join(path, 'data.csv')
            trials_path = os.path.join(path, 'trials.csv')
            behavior_columns_path = os.path.join(path, 'behavior_columns.yml')
            self.spike_train = np.load(spike_train_path, allow_pickle=True)
            with open(bin_size_path, 'r', encoding='utf-8') as f:
                self.bin_size = float(f.read())
            self.data = pd.read_csv(data_path, index_col=0, header=0)
            self.trials = pd.read_csv(trials_path, header=0)
            with open(behavior_columns_path, 'r', encoding='utf-8') as f:
                self.behavior_columns = dict(yaml2dict(str(f.read())))
            self.logger.info(f"Load image from {path}")
