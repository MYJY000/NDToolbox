import os
import torch
import joblib
import time
from copy import deepcopy
from torch.optim import lr_scheduler

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
        self.net = None

    def fit(self, x, y):
        raise NotImplementedError

    def predict(self, x=None):
        self.net.eval()
        if x is not None:
            self.feed_data([x])
        with torch.no_grad():
            self.output = self.net(self.data_x)
        self.net.train()
        return self.output

    def load(self, path):
        raise NotImplementedError

    def save(self, path, epoch=-1, cur_iter=-1):
        raise NotImplementedError

    def validation(self, x, y_true, metric_list):
        y_pred = self.predict(x)
        metric_dict = build_metric(y_true, y_pred, metric_list)
        self.logger.info(f"Model {self.name} metric:")
        for metric_name, metric_value in metric_dict.items():
            self.logger.info(f"{metric_name} - {metric_value}")
        return metric_dict

    def init_train_setting(self, train_opt):
        raise NotImplementedError

    def optimize_parameters(self, cur_iter):
        raise NotImplementedError

    def feed_data(self, data):
        self.data_x = torch.tensor(data[0], dtype=torch.float32).to(self.device)
        if len(data) > 1:
            self.data_y = torch.tensor(data[1], dtype=torch.float32).to(self.device)

    def save_network(self, nets, network_path, net_label, cur_iter, net_keys='net'):
        if cur_iter == -1:
            cur_iter = 'latest'
        save_filename = f'{net_label}_{cur_iter}.pth'
        save_path = os.path.join(network_path, save_filename)
        os.makedirs(network_path, exist_ok=True)

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

    def load_network(self, net, load_path, strict=True, net_key='net'):
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
        self.logger.info(f'Network: {net_cls_str}, with parameters: {net_params:,d}')
        self.logger.info(net_str)

    def save_training_state(self, state_path, epoch, cur_iter):
        if cur_iter == -1:
            return
        state = {'epoch': epoch, 'iter': cur_iter, 'optimizers': [], 'schedulers': []}
        for o in self.optimizers:
            state['optimizers'].append(o.state_dict())
        for s in self.schedulers:
            state['schedulers'].append(s.state_dict())
        save_filenames = f'{cur_iter}.state'
        save_path = os.path.join(state_path, save_filenames)
        os.makedirs(state_path, exist_ok=True)

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

    def resume_training(self, resume_state):
        resume_optimizers = resume_state['optimizers']
        resume_schedulers = resume_state['schedulers']
        assert len(resume_optimizers) == len(self.optimizers), 'Wrong number of optimizers'
        assert len(resume_schedulers) == len(self.schedulers), 'Wrong number of schedulers'
        for i, o in enumerate(resume_optimizers):
            self.optimizers[i].load_state_dict(o)
        for i, s in enumerate(resume_schedulers):
            self.schedulers[i].load_state_dict(s)

    def update_lr(self, cur_iter):
        if cur_iter > 1:
            for scheduler in self.schedulers:
                scheduler.step()

    def get_current_lr(self):
        return [param_group['lr'] for param_group in self.optimizers[0].param_groups]

    def get_optimizer(self, optim_type, params, lr, **kwargs):
        if optim_type == 'Adam':
            optimizer = torch.optim.Adam(params, lr, **kwargs)
        elif optim_type == 'AdamW':
            optimizer = torch.optim.AdamW(params, lr, **kwargs)
        elif optim_type == 'Adamax':
            optimizer = torch.optim.Adamax(params, lr, **kwargs)
        elif optim_type == 'SGD':
            optimizer = torch.optim.SGD(params, lr, **kwargs)
        elif optim_type == 'ASGD':
            optimizer = torch.optim.ASGD(params, lr, **kwargs)
        elif optim_type == 'RMSprop':
            optimizer = torch.optim.RMSprop(params, lr, **kwargs)
        elif optim_type == 'Rprop':
            optimizer = torch.optim.Rprop(params, lr, **kwargs)
        else:
            raise NotImplementedError(f'optimizer {optim_type} is not supported.')
        return optimizer

    def setup_optimizers(self, train_opt):
        optim_params = []
        for k, v in self.net.named_parameters():
            if v.requires_grad:
                optim_params.append(v)
            else:
                self.logger.warning(f'Parameter {k} will not be optimized.')

        optim_type = train_opt['optimizer'].pop('type')
        self.optimizer_net = self.get_optimizer(optim_type, optim_params, **train_opt['optimizer'])
        self.optimizers.append(self.optimizer_net)

    def setup_schedulers(self, train_opt):
        scheduler_type = train_opt['scheduler'].pop('type')
        if scheduler_type == 'LambdaLR':
            for optimizer in self.optimizers:
                self.schedulers.append(lr_scheduler.LambdaLR(optimizer, **train_opt['scheduler']))
        elif scheduler_type == 'MultiplicativeLR':
            for optimizer in self.optimizers:
                self.schedulers.append(lr_scheduler.MultiplicativeLR(optimizer, **train_opt['scheduler']))
        elif scheduler_type == 'StepLR':
            for optimizer in self.optimizers:
                self.schedulers.append(lr_scheduler.StepLR(optimizer, **train_opt['scheduler']))
        elif scheduler_type == 'MultiStepLR':
            for optimizer in self.optimizers:
                self.schedulers.append(lr_scheduler.MultiStepLR(optimizer, **train_opt['scheduler']))
        elif scheduler_type == 'ConstantLR':
            for optimizer in self.optimizers:
                self.schedulers.append(lr_scheduler.ConstantLR(optimizer, **train_opt['scheduler']))
        elif scheduler_type == 'LinearLR':
            for optimizer in self.optimizers:
                self.schedulers.append(lr_scheduler.LinearLR(optimizer, **train_opt['scheduler']))
        elif scheduler_type == 'ExponentialLR':
            for optimizer in self.optimizers:
                self.schedulers.append(lr_scheduler.ExponentialLR(optimizer, **train_opt['scheduler']))
        elif scheduler_type == 'PolynomialLR':
            for optimizer in self.optimizers:
                self.schedulers.append(lr_scheduler.PolynomialLR(optimizer, **train_opt['scheduler']))
        elif scheduler_type == 'CosineAnnealingLR':
            for optimizer in self.optimizers:
                self.schedulers.append(lr_scheduler.CosineAnnealingLR(optimizer, **train_opt['scheduler']))
        else:
            raise NotImplementedError(f'Scheduler type {scheduler_type} is not supported.')
