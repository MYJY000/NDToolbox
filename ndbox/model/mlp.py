import numpy as np

from .base_model import DLBaseModel
from ndbox.utils import MODEL_REGISTRY


@MODEL_REGISTRY.register()
class MLPRegression(DLBaseModel):
    def __init__(self, **kwargs):
        super(MLPRegression, self).__init__()

