import os
import joblib

from ndbox.utils import get_root_logger
from ndbox.metric import calculate_metric


class MLBaseModel:
    """
    Machine learning base model.
    """

    def __init__(self, **kwargs):
        self.logger = get_root_logger()
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
        self.logger.info(f"Loading {self.__class__.__name__} model from '{path}'.")

    def save(self, path):
        self.logger.info(f"Save {self.__class__.__name__} model in '{path}'.")
        joblib.dump(self.model, path)

    def validation(self, x, y_true, metric_list):
        y_pred = self.predict(x)
        metric_dict = {}
        for metric_name, metric_opt in metric_list.items():
            metric_value = calculate_metric(y_true, y_pred, metric_opt)
            metric_dict[metric_name] = metric_value
            self.logger.info(f"Model {self.__class__.__name__} metric {metric_name}: {metric_value}")
        return metric_dict

    def get_params(self):
        return self.params
