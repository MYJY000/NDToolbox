import os
import torch
import joblib
import time
from copy import deepcopy

from ndbox.utils import get_root_logger
from ndbox.metric import build_metric


class MLBaseModel:
    """
    Machine learning base model.
    """

    def __init__(self, **kwargs):
        self.logger = get_root_logger()
        self.identifier = 'ML'
        self.name = self.__class__.__name__
        self.params = {}
        self.model = None

    def fit(self, x, y):
        raise NotImplementedError

    def predict(self, x):
        raise NotImplementedError

    def load(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"'{path}' File not found!")
        self.model = joblib.load(path)
        self.logger.info(f"Loading {self.name} model from {path}.")

    def save(self, path):
        self.logger.info(f"Save {self.name} model in {path}.")
        joblib.dump(self.model, path)

    def validation(self, x, y_true, metric_list):
        y_pred = self.predict(x)
        metric_dict = build_metric(y_true, y_pred, metric_list)
        self.logger.info(f"Model {self.name} metric:")
        for metric_name, metric_value in metric_dict.items():
            self.logger.info(f"{metric_name} - {metric_value}")
        return metric_dict

    def get_params(self):
        return self.params


class DLBaseModel:
    """
    Deep learning base model.

    Modified from:
    https://github.com/XPixelGroup/BasicSR/blob/master/basicsr/models/base_model.py
    """

    def __init__(self, device='cpu', **kwargs):
        self.logger = get_root_logger()
        self.identifier = 'DL'
        self.name = self.__class__.__name__

        self.device = torch.device(device)
        self.optimizers = []
        self.schedulers = []

    def fit(self, x, y):
        raise NotImplementedError

    def predict(self, x):
        raise NotImplementedError

    def load(self, path):
        raise NotImplementedError

    def save(self, path):
        raise NotImplementedError

    def validation(self, x, y_true, metric_list):
        raise NotImplementedError

    def feed_data(self, data):
        raise NotImplementedError

    def init_train_setting(self):
        raise NotImplementedError

    def save_network(self, nets, model_path, net_label, epoch, net_keys='params'):
        save_filename = f'{net_label}_{epoch}.pth'
        save_path = os.path.join(model_path, save_filename)

        nets = nets if isinstance(nets, list) else [nets]
        net_keys = net_keys if isinstance(net_keys, list) else [net_keys]
        assert len(nets) == len(net_keys), 'The length of net and key must match'

        save_dict = {}
        for net, net_key in zip(nets, net_keys):
            state_dict = net.state_dict()
            for key, param in state_dict.items():
                if key.startswith('module.'):
                    key = key[7:]
                state_dict[key] = param.cpu()
            save_dict[net_key] = state_dict

        retry = 3
        while retry > 0:
            try:
                torch.save(save_dict, save_path)
            except Exception as e:
                self.logger.warning(f'Save model error: {e}, remaining'
                                    f' retry times: {retry - 1}')
                time.sleep(1)
            else:
                break
            finally:
                retry -= 1
        if retry == 0:
            self.logger.warning(f'Still cannot save {save_path}')

    def load_network(self, net, load_path, strict=True, net_key='params'):
        load_net = torch.load(load_path, map_location=lambda storage, loc: storage)
        load_net = load_net[net_key]
        self.logger.info(f'Loading {net.__class__.__name__} network from {load_path}')
        for key, param in deepcopy(load_net).items():
            if key.startswith('module.'):
                load_net[key[7:]] = param
                load_net.pop(key)
        self._print_different_keys(net, load_net, strict)
        net.load_state_dict(load_net, strict=strict)

    def _print_different_keys(self, net, load_net, strict=True):
        net = net.state_dict()
        net_keys = set(net.keys())
        load_net_keys = set(load_net.keys())

        if net_keys != load_net_keys:
            self.logger.warning('Current net - loaded net:')
            for v in sorted(list(net_keys - load_net_keys)):
                self.logger.warning(f'  {v}')
            self.logger.warning('Loaded net - current net:')
            for v in sorted(list(load_net_keys - net_keys)):
                self.logger.warning(f'  {v}')

        if not strict:
            common_keys = net_keys & load_net_keys
            for k in common_keys:
                if net[k].size() != load_net[k].size():
                    self.logger.warning(f'Size different, ignore [{k}], Current net: '
                                        f'{net[k].shape}, loaded net: {load_net[k].shape}')
                    load_net[k + '.ignore'] = load_net.pop(k)

    def print_network(self, net):
        net_cls_str = f'{net.__class__.__name__}'
        net_str = str(net)
        net_params = sum(map(lambda x: x.numel(), net.parameters()))
        self.logger.info(f'Network: {net_cls_str}, with parameters: {net_params:, d}')
        self.logger(net_str)

    def save_training_state(self, state_path, epoch):
        state = {'epoch': epoch, 'optimizers': [], 'schedulers': []}
        for o in self.optimizers:
            state['optimizers'].append(o.state_dict())
        for s in self.schedulers:
            state['schedulers'].append(s.state_dict())
        save_filenames = f'{epoch}.state'
        save_path = os.path.join(state_path, save_filenames)

        retry = 3
        while retry > 0:
            try:
                torch.save(state, save_path)
            except Exception as e:
                self.logger.warning(f'Save training state error: {e}, '
                                    f'remaining retry times: {retry - 1}')
                time.sleep(1)
            else:
                break
            finally:
                retry -= 1
        if retry == 0:
            self.logger.warning(f'Still cannot saving {save_path}.')

    def resume_training(self, resume_path):
        if torch.cuda.is_available():
            device_id = torch.cuda.current_device()
            resume_state = torch.load(resume_path, map_location=lambda storage, loc: storage.cuda(device_id))
        else:
            resume_state = torch.load(resume_path, map_location=lambda storage, loc: storage)
        self.logger.info(f"Resume training from epoch: {resume_state['epoch']}")
        '''
        修改参数为 resume_state，将上面部分代码置入流程中
        '''

    def update_lr(self, current_iter):
        if current_iter > 1:
            for scheduler in self.schedulers:
                scheduler.step()
