import numpy as np

from .base_model import DLBaseModel
from .components import build_arch
from ndbox.utils import MODEL_REGISTRY


@MODEL_REGISTRY.register()
class MLPRegression(DLBaseModel):

    def __init__(self, network, **kwargs):
        super(MLPRegression, self).__init__(**kwargs)
        self.net = build_arch(network)
        self.net = self.net.to(self.device)
        self.print_network(self.net)

    def init_train_setting(self):
        pass
