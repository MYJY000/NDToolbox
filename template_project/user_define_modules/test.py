from sklearn.linear_model import LogisticRegression

from ndbox.model.base_model import MLBaseModel
from ndbox.utils import MODEL_REGISTRY


@MODEL_REGISTRY.register()
class LogisticModel(MLBaseModel):

    def __init__(self, penalty='l2', **kwargs):
        super(LogisticModel, self).__init__()
        self.params['penalty'] = penalty

    def fit(self, x, y):
        self.model = LogisticRegression(penalty=self.params['penalty'])
        self.model.fit(x, y)

    def predict(self, x):
        y_pred = self.model.predict(x)
        return y_pred
