import os

import joblib
from sklearn import linear_model

from ndbox.utils import MODEL_REGISTRY
from .base_model import MLBaseModel


@MODEL_REGISTRY.register()
class WienerFilterRegression(MLBaseModel):
    """
    Class for the Wiener Filter Decoder. There are no parameters
    to set. This simply leverages the scikit-learn linear regression.
    """

    def __init__(self, **kwargs):
        super(WienerFilterRegression, self).__init__()

    def fit(self, x, y):
        self.model = linear_model.LinearRegression()
        self.model.fit(x, y)

    def predict(self, x):
        y_pred = self.model.predict(x)
        return y_pred


@MODEL_REGISTRY.register()
class WienerFilterClassification(MLBaseModel):
    """
    Class for the Wiener Filter Decoder. There are no parameters to set.
    This simply leverages the scikit-learn logistic regression.

    :param C: Inverse of regularization strength; must be a positive float.
    """

    def __init__(self, C=1, **kwargs):
        super(WienerFilterClassification, self).__init__()
        self.params['C'] = C

    def fit(self, x, y):
        """
        Train Wiener Filter Decoder.

        :param x: numpy 2d array of shape [n_samples, n_features]
            This is the neural datasets.
        :param y: numpy 2d array of shape [n_samples, n_outputs]
            This is the outputs that are being predicted.
        """

        self.model = linear_model.LogisticRegression(C=self.params['C'], multi_class='auto')
        self.model.fit(x, y)

    def predict(self, x):
        """
        Predict outcomes using trained Wiener Cascade Decoder.

        :param x: numpy 2d array of shape [n_samples, n_features]
            This is the neural datasets being used to predict outputs.
        :return: numpy 2d array of shape [n_samples, n_outputs]
            The predicted outputs.
        """

        y_pred = self.model.predict(x)
        return y_pred

    def load(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"'{path}' File not found!")
        self.model = joblib.load(path)
        self.params['C'] = self.model.C
        self.logger.info(f"Loading {self.__class__.__name__} model from '{path}'.")
