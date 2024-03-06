from .base_model import DLBaseModel
from .components import build_arch, build_loss
from ndbox.utils import MODEL_REGISTRY


@MODEL_REGISTRY.register()
class MLPRegression(DLBaseModel):

    def __init__(self, network, path=None, **kwargs):
        super(MLPRegression, self).__init__(**kwargs)
        self.net = build_arch(network)
        self.net = self.net.to(self.device)
        self.print_network(self.net)

        if path is not None:
            pretrain_net_path = path.get('pretrain_net_path')
            if pretrain_net_path is not None:
                net_key = path.get('net_key', 'net')
                strict_load = path.get('strict_load', True)
                self.load_network(self.net, pretrain_net_path, strict_load, net_key)

    def init_train_setting(self, train_opt):
        self.net.train()
        self.loss = build_loss(train_opt['loss'])
        self.setup_optimizers(train_opt)
        self.setup_schedulers(train_opt)

    def optimize_parameters(self, cur_iter):
        self.optimizer_net.zero_grad()
        self.output = self.net(self.data_x)
        loss = self.loss(self.output, self.data_y)
        loss.backward()
        self.optimizer_net.step()

    def save(self, path, epoch=-1, cur_iter=-1):
        self.save_network(self.net, path, 'mlp', cur_iter, 'net')
        self.save_training_state(path, epoch, cur_iter)
