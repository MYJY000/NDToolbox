import os
import joblib

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

    def __init__(self, **kwargs):
        self.logger = get_root_logger()
        self.identifier = 'DL'
        self.name = self.__class__.__name__
