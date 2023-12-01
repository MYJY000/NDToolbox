from sklearn.linear_model import LinearRegression

from ndbox.model.base_model import MLBaseModel
from ndbox.utils import MODEL_REGISTRY


@MODEL_REGISTRY.register()
class LinearRegressionModel(MLBaseModel):

    def __init__(self, penalty='l2', **kwargs):
        super(LinearRegressionModel, self).__init__()
        self.params['penalty'] = penalty

    def fit(self, x, y):
        self.model = LinearRegression()
        self.model.fit(x, y)

    def predict(self, x):
        y_pred = self.model.predict(x)
        return y_pred
