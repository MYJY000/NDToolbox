from torch import nn as nn

from ndbox.utils import ARCH_REGISTRY


@ARCH_REGISTRY.register()
class MLP(nn.Module):

    def __init__(self, dim_in, dim_out, hidden_dims=None, dropout=0, **kwargs):
        super(MLP, self).__init__()
        if hidden_dims is None:
            hidden_dims = []
        if not isinstance(hidden_dims, list):
            hidden_dims = [hidden_dims]
        layers = []
        if len(hidden_dims) == 0:
            layers.append(nn.Linear(dim_in, dim_out))
        else:
            prev_dim = dim_in
            for layer_dim in hidden_dims:
                layers.append(nn.Linear(prev_dim, layer_dim))
                layers.append(nn.ReLU(inplace=True))
                if dropout > 0:
                    layers.append(nn.Dropout(p=dropout))
                prev_dim = layer_dim
            layers.append(nn.Linear(prev_dim, dim_out))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.layers(x)
        return x
