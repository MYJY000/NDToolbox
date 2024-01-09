import os
import joblib
import numpy as np
from tqdm import trange
from sklearn import linear_model

from ndbox.utils import files_form_folder, MODEL_REGISTRY
from .base_model import MLBaseModel


@MODEL_REGISTRY.register()
class WienerCascadeRegression(MLBaseModel):
    """
    Class for the Wiener Cascade Decoder.

    :param degree: integer, optional, default 3.
        The degree of the polynomial used for the static nonlinear
    """

    def __init__(self, degree: int = 3, **kwargs):
        super(WienerCascadeRegression, self).__init__()
        self.params['degree'] = degree

    def fit(self, x, y):
        """
        Train Wiener Cascade Decoder.

        :param x: numpy 2d array of shape [n_samples, n_features]
            This is the neural datasets.
        :param y: numpy 2d array of shape [n_samples, n_outputs]
            This is the outputs that are being predicted
        """

        num_outputs = y.shape[1]
        models = {}
        with trange(num_outputs) as t:
            for i in t:
                t.set_description(f"Train Wiener Cascade Regression")
                reg = linear_model.LinearRegression()
                reg.fit(x, y[:, i])
                y_pred = reg.predict(x)
                poly = np.polyfit(y_pred, y[:, i], self.params['degree'])
                models['r_' + str(i)] = reg
                models['p_' + str(i)] = poly
        self.model = models

    def predict(self, x):
        """
        Predict outcomes using trained Wiener Cascade Decoder.

        :param x: numpy 2d array of shape [n_samples, n_features]
            This is the neural datasets being used to predict outputs.
        :return: numpy 2d array of shape [n_samples, n_outputs]
            The predicted outputs.
        """

        num_outputs = len(self.model) // 2
        y_pred = np.empty([x.shape[0], num_outputs])
        for i in range(num_outputs):
            reg, poly = self.model['r_' + str(i)], self.model['p_' + str(i)]
            y_pred_linear = reg.predict(x)
            y_pred[:, i] = np.polyval(poly, y_pred_linear)
        return y_pred

    def load(self, path):
        """
        Load model.

        :param path: str. The path of models to be loaded.
        """

        reg_files = files_form_folder(path, 'r_*.pkl')
        poly_files = files_form_folder(path, 'p_*.pkl')
        if len(reg_files) != len(poly_files):
            raise ValueError(f"The number of linear regression model files is not "
                             f"equal to that of polynomial regression model files!"
                             f" The former is [{len(reg_files)}] and the latter is"
                             f" [{len(poly_files)}].")
        num_outputs = len(reg_files)
        models = {}
        with trange(num_outputs) as t:
            for i in t:
                t.set_description(f"Load Model")
                models['r_' + str(i)] = joblib.load(
                    os.path.join(path, 'r_' + str(i) + '.pkl')
                )
                models['p_' + str(i)] = joblib.load(
                    os.path.join(path, 'p_' + str(i) + '.pkl')
                )
                self.params['degree'] = len(models['p_' + str(i)])
        self.model = models
        self.logger.info(f"Loading {self.name} model from '{path}'")

    def save(self, path):
        """
        Save model.

        :param path: str. The path of models to be saved.
        """

        if not os.path.exists(path):
            os.makedirs(path)
        for filename, model in self.model.items():
            joblib.dump(model, os.path.join(path, filename + '.pkl'))
