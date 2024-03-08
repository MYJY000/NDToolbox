from torch import nn as nn

from ndbox.utils import ARCH_REGISTRY


@ARCH_REGISTRY.register()
class LSTM(nn.Module):

    def __init__(self, input_size, output_size, hidden_size, seq_len, num_layers=1, dropout=0, **kwargs):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True
        )
        self.linear = nn.Linear(hidden_size * seq_len, output_size)

    def forward(self, x):
        x, _ = self.lstm(x)
        batch_size, seq_len, hidden_size = x.shape
        x = x.reshape(batch_size, hidden_size * seq_len)
        x = self.linear(x)
        x = x.reshape(batch_size, -1)
        return x
