from torch import nn as nn
from torch.nn import init as init
from torch.utils.data import DataLoader, Dataset


class PairedDataset(Dataset):

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return len(self.x)


class DatasetIter:

    def __init__(self, loader):
        self.loader = loader
        self.iterator = iter(loader)

    def next(self):
        try:
            return next(self.iterator)
        except StopIteration:
            return None

    def reset(self):
        self.iterator = iter(self.loader)


def get_paired_dataloader(x, y):
    return DataLoader(PairedDataset(x, y), batch_size=1, shuffle=False)


def make_layer(basic_block, nums, **kwargs):
    layers = []
    for _ in range(nums):
        layers.append(basic_block(**kwargs))
    return nn.Sequential(*layers)


def default_init_weights(module_list, scale=1, bias_fill=0, **kwargs):
    if not isinstance(module_list, list):
        module_list = [module_list]
    for module in module_list:
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)
