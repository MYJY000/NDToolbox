import os
import numpy as np
from numpy.linalg import inv

from .base_model import MLBaseModel
from ndbox.utils import MODEL_REGISTRY


@MODEL_REGISTRY.register()
class KalmanFilterRegression(MLBaseModel):
    """
    Class for the Kalman Filter Decoder.

    :param C: float, optional, default 1.
        This parameter scales the noise matrix associated with
        the transition in kinematic states. It effectively
        allows changing the weight of the new neural evidence
        in the current update.
    """

    def __init__(self, C: float = 1, **kwargs):
        super(KalmanFilterRegression, self).__init__()
        self.params['C'] = C

    def fit(self, x, y):
        """
        Train Kalman Filter Decoder.

        :param x: numpy 2d array of shape [n_samples(i.e. bins) , n_neurons]
            This is the neural datasets in Kalman filter format.
        :param y: numpy 2d array of shape [n_samples(i.e. bins), n_outputs]
            This is the outputs that are being predicted.
        """

        _X = np.matrix(y.T)
        Z = np.matrix(x.T)
        nt = _X.shape[1]
        x2 = _X[:, 1:]
        x1 = _X[:, 0:nt - 1]
        A = x2 * x1.T * inv(x1 * x1.T)
        W = (x2 - A * x1) * (x2 - A * x1).T / (nt - 1) / self.params['C']
        H = Z * _X.T * (inv(_X * _X.T))
        Q = ((Z - H * _X) * (Z - H * _X).T) / nt
        params = [A, W, H, Q]
        self._X = _X
        self.model = params

    def predict(self, x):
        """
        Predict outcomes using trained Kalman Filter Decoder.

        :param x: numpy 2d array of shape [n_samples(i.e. bins), n_neurons]
            This is the neural datasets in Kalman filter format.
        :return: numpy 2d array of shape [n_samples(i.e. bins), n_outputs]
            The predicted outputs.
        """

        A, W, H, Q = self.model
        _X = self._X
        Z = np.matrix(x.T)
        num_states = _X.shape[0]
        states = np.empty(_X.shape)
        P = np.matrix(np.zeros([num_states, num_states]))
        state = _X[:, 0]
        states[:, 0] = np.copy(np.squeeze(state))

        for t in range(_X.shape[1] - 1):
            P_m = A * P * A.T + W
            state_m = A * state
            K = P_m * H.T * inv(H * P_m * H.T + Q)
            P = (np.matrix(np.eye(num_states)) - K * H) * P_m
            state = state_m + K * (Z[:, t + 1] - H * state_m)
            states[:, t + 1] = np.squeeze(state)
        y_pred = states.T
        return y_pred

    def load(self, path):
        """
        Load model.

        :param path: str. The path of models to be loaded.
        """

        if not os.path.exists(path):
            path = path + '.npz'
            if not os.path.exists(path):
                raise FileNotFoundError(f"'{path}' File not found!")
        data = np.load(path)
        self.model = [data['A'], data['W'], data['H'], data['Q']]
        self.params['C'] = data['C'][0]
        self.logger.info(f"Loading {self.__class__.__name__} model from '{path}'")

    def save(self, path):
        """
        Save model.

        :param path: str. The path of models to be saved.
        """

        self.logger.info(f"Save {self.__class__.__name__} model in {path}.")
        A, W, H, Q = self.model
        C = np.array([self.params['C']])
        np.savez(path, A=A, W=W, H=H, Q=Q, C=C)
