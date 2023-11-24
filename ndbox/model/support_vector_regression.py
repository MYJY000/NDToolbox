import os

import joblib
import numpy as np
from sklearn.svm import SVR
from tqdm import trange

from ndbox.utils import files_form_folder, MODEL_REGISTRY
from .base_model import MLBaseModel


@MODEL_REGISTRY.register()
class SupportVectorRegression(MLBaseModel):
    """
    Class for the Support Vector Regression (SVR) Decoder
    This simply leverages the scikit-learn SVR.

    :param max_iter: integer, default=-1
        the maximum number of iterations to run (to save time)
        max_iter=-1 means no limit.
    :param C: float, default=3.0
        Penalty parameter of the error term.
    """

    def __init__(self, max_iter=-1, C=3.0, **kwargs):
        super(SupportVectorRegression, self).__init__()
        self.params['max_iter'] = max_iter
        self.params['C'] = C

    def fit(self, x, y):
        num_outputs = y.shape[1]
        models = {}
        with trange(num_outputs) as t:
            for i in t:
                t.set_description(f"Train Support Vector Regression")
                model = SVR(max_iter=self.params['max_iter'], C=self.params['C'])
                model.fit(x, y[:, i])
                models['svr_' + str(i)] = model
        self.model = models

    def predict(self, x):
        num_outputs = len(self.model)
        y_pred = np.empty([x.shape[0], num_outputs])
        for i in range(num_outputs):
            model = self.model['svr_' + str(i)]
            y_pred[:, i] = model.predict(x)
        return y_pred

    def load(self, path):
        svr_files = files_form_folder(path, 'svr_*.pkl')
        num_outputs = len(svr_files)
        if num_outputs == 0:
            raise FileNotFoundError(f"Not found file in '{path}' with format 'svr_*.pkl'.")
        models = {}
        with trange(num_outputs) as t:
            for i in t:
                t.set_description(f"Load Model")
                models['svr_' + str(i)] = joblib.load(
                    os.path.join(path, 'svr_' + str(i) + '.pkl')
                )
                self.params['max_iter'] = models['svr_' + str(i)].max_iter
                self.params['C'] = models['svr_' + str(i)].C
        self.model = models
        self.logger.info(f"Loading {self.__class__.__name__} model from '{path}'")

    def save(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
        for filename, model in self.model.items():
            joblib.dump(model, os.path.join(path, filename + '.pkl'))
