import os
import torch
import joblib
import time

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
        self.train_x = data.get('train_x')
        self.train_y = data.get('train_y')
        self.train_opt = data.get('train_opt')
        self.val_x = data.get('val_x')
        self.val_y = data.get('val_y')
        self.val_opt = data.get('val_opt')
        self.model_path = data.get('model_path')

    def save_state(self, epoch):
        state = {
            'epoch': epoch,
            'optimizers': [],
            'schedulers': []
        }
        for optimizer in self.optimizers:
            state['optimizers'].append(optimizer.state_dict())
        for scheduler in self.schedulers:
            state['schedulers'].append(scheduler.state_dict())
        save_filename = f'{epoch}.state'
        save_path = os.path.join(self.model_path, save_filename)

        retry = 3
        while retry > 0:
            try:
                torch.save(state, save_path)
            except Exception as e:
                self.logger.warning(f'Save training state error: {e},'
                                    f' remaining retry times: {retry - 1}')
                time.sleep(1)
            else:
                break
            finally:
                retry -= 1
        if retry == 0:
            self.logger.warning(f'Still cannot save {save_path}. Just ignore it.')

    def save_network(self, net, net_label, epoch, key='net'):
        save_filename = f'{net_label}_{epoch}.pth'
        save_path = os.path.join(self.model_path, save_filename)

        net = net if isinstance(net, list) else [net]
        key = key if isinstance(key, list) else [key]
        assert len(net) == len(key), 'The length of net and key must match'

        save_dict = {}
        for net_, key_ in zip(net, key):
            pass

    def load_network(self, net, net_label, epoch):
        pass
