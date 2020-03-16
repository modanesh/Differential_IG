import torch
import torch.nn as nn
from functions import TernaryTanh
from torch.autograd import Variable
import tools as tl

class HxQBNet(nn.Module):
    """
    Quantized Bottleneck Network(QBN) for hidden states of GRU
    """

    def __init__(self, input_size, x_features):
        super(HxQBNet, self).__init__()
        self.bhx_size = x_features
        f1, f2 = int(8 * x_features), int(4 * x_features)
        self.encoder = nn.Sequential(nn.Linear(input_size, f1),
                                     nn.Tanh(),
                                     nn.Linear(f1, f2),
                                     nn.Tanh(),
                                     nn.Linear(f2, x_features),
                                     TernaryTanh())

        self.decoder = nn.Sequential(nn.Linear(x_features, f2),
                                     nn.Tanh(),
                                     nn.Linear(f2, f1),
                                     nn.Tanh(),
                                     nn.Linear(f1, input_size),
                                     nn.Tanh())

    def forward(self, x):
        x = self.encode(x)
        return self.decode(x), x

    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        return self.decoder(x)


class MMNet(nn.Module):
    """
    Moore Machine Network(MMNet) definition.
    """
    def __init__(self, net, hx_qbn=None, obs_qbn=None):
        super(MMNet, self).__init__()
        self.bhx_units = hx_qbn.bhx_size if hx_qbn is not None else None
        self.gru_units = net.gru_units
        self.obx_net = obs_qbn
        self.gru_net = net
        self.bhx_net = hx_qbn
        self.actor_linear = self.gru_net.get_action_linear

    def init_hidden(self, batch_size=1):
        return self.gru_net.init_hidden(batch_size)

    def forward(self, x, inspect=False):
        if inspect:
            x, hx = x
            critic, actor, hx, (ghx, bhx, input_c, input_x, input_tanh) = self.gru_net((x, hx), input_fn=self.obx_net, hx_fn=self.bhx_net, inspect=True)
            return critic, actor, hx, (ghx, bhx), (input_c, input_x, input_tanh)
        else:
            input_c = self.gru_net(x, input_fn=self.obx_net, hx_fn=self.bhx_net, inspect=False)
            return input_c

    def get_action_linear(self, state, decode=False):
        if decode:
            hx = self.bhx_net.decode(state)
        else:
            hx = state
        return self.actor_linear(hx)

    def transact(self, o_x, hx_x):
        hx_x = self.gru_net.transact(self.obx_net.decode(o_x), self.bhx_net.decode(hx_x))
        _, hx_x = self.bhx_net(hx_x)
        return hx_x

    def state_encode(self, state):
        return self.bhx_net.encode(state)

    def obs_encode(self, obs, hx=None):
        if hx is None:
            hx = Variable(torch.zeros(1, self.gru_units))
            if next(self.parameters()).is_cuda:
                hx = hx.cuda()
        _, _, _, (_, _, _, input_x) = self.gru_net((obs, hx), input_fn=self.obx_net, hx_fn=self.bhx_net, inspect=True)
        return input_x


class ObsQBNet(nn.Module):
    """
    Quantized Bottleneck Network(QBN) for observation features.
    """

    def __init__(self, input_size, x_features):
        super(ObsQBNet, self).__init__()
        self.bhx_size = x_features
        f1 = int(8 * x_features)
        self.encoder = nn.Sequential(nn.Linear(input_size, f1),
                                     nn.Tanh(),
                                     nn.Linear(f1, x_features),
                                     TernaryTanh())

        self.decoder = nn.Sequential(nn.Linear(x_features, f1),
                                     nn.Tanh(),
                                     nn.Linear(f1, input_size),
                                     nn.ReLU6())

    def forward(self, x):
        encoded, before_ttanh = self.encode(x)
        decoded = self.decode(encoded)
        return decoded, encoded, before_ttanh

    def encode(self, x):
        linear1 = self.encoder[0](x)
        tanh = self.encoder[1](linear1)
        linear2 = self.encoder[2](tanh)
        ttanh = self.encoder[3](linear2)
        return self.encoder(x), linear2

    def decode(self, x):
        return self.decoder(x)


class GRUNet(nn.Module):
    """
    Gated Recurrent Unit Network(GRUNet) definition.
    """

    def __init__(self, input_size, gru_cells, total_actions):
        super(GRUNet, self).__init__()
        self.gru_units = gru_cells
        self.noise = False
        self.conv1 = nn.Conv2d(input_size, 32, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 16, 3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(16, 8, 3, stride=2, padding=1)

        self.input_ff = nn.Sequential(self.conv1, nn.ReLU(),
                                      self.conv2, nn.ReLU(),
                                      self.conv3, nn.ReLU(),
                                      self.conv4, nn.ReLU6())
        self.input_c_features = 8 * 5 * 5
        self.input_c_shape = (8, 5, 5)
        self.gru = nn.GRUCell(self.input_c_features, gru_cells)

        self.critic_linear = nn.Linear(gru_cells, 1)
        self.actor_linear = nn.Linear(gru_cells, total_actions)

        self.apply(tl.weights_init)
        self.actor_linear.weight.data = tl.normalized_columns_initializer(self.actor_linear.weight.data, 0.01)
        self.actor_linear.bias.data.fill_(0)
        self.critic_linear.weight.data = tl.normalized_columns_initializer(self.critic_linear.weight.data, 1.0)
        self.critic_linear.bias.data.fill_(0)

        self.gru.bias_ih.data.fill_(0)
        self.gru.bias_hh.data.fill_(0)

    def forward(self, input, input_fn=None, hx_fn=None, inspect=False):
        if inspect:
            input, hx = input
            c_input = self.input_ff(input)
            c_input = c_input.view(-1, self.input_c_features)
            input, input_x, linear2 = input_fn(c_input) if input_fn is not None else (c_input, c_input)
            ghx = self.gru(input, hx)
            hx, bhx = hx_fn(ghx) if hx_fn is not None else (ghx, ghx)
            return self.critic_linear(hx), self.actor_linear(hx), hx, (ghx, bhx, c_input, input_x, linear2)
        else:
            c_input = self.input_ff(input)
            c_input = c_input.view(-1, self.input_c_features)
            input, input_x, linear2 = input_fn(c_input) if input_fn is not None else (c_input, c_input)
            return c_input, input_x, linear2

    def init_hidden(self, batch_size=1):
        return torch.zeros(batch_size, self.gru_units)

    def get_action_linear(self, state):
        return self.actor_linear(state)

    def transact(self, o_x, hx):
        hx = self.gru(o_x, hx)
        return hx


class ControlGRUNet(nn.Module):
    """
    Gated Recurrent Unit Network(GRUNet) definition
    """
    def __init__(self, input_size, gru_cells, total_actions):
        super(ControlGRUNet, self).__init__()
        self.gru_units = gru_cells
        self.noise = False

        self.input_ff = nn.Sequential(nn.Linear(input_size, 16),
                                      nn.ELU(),
                                      nn.Linear(16, 8),
                                      nn.ReLU6())
        self.input_flat_size = 8
        self.input_c_features = self.input_flat_size
        self.gru = nn.GRUCell(self.input_flat_size, 32)

        self.critic_linear = nn.Linear(gru_cells, 1)
        self.actor_linear = nn.Linear(gru_cells, total_actions)

        self.apply(tl.weights_init)
        self.actor_linear.weight.data = tl.normalized_columns_initializer(self.actor_linear.weight.data, 0.01)
        self.actor_linear.bias.data.fill_(0)
        self.critic_linear.weight.data = tl.normalized_columns_initializer(self.critic_linear.weight.data, 1.0)
        self.critic_linear.bias.data.fill_(0)

        self.gru.bias_ih.data.fill_(0)
        self.gru.bias_hh.data.fill_(0)

    def forward(self, input, input_fn=None, hx_fn=None, inspect=False):
        if inspect:
            input, hx = input
            c_input = self.input_ff(input)
            c_input = c_input.view(-1, self.input_flat_size)
            input, input_x, input_tanh = input_fn(c_input) if input_fn is not None else (c_input, c_input, None)
            ghx = self.gru(input, hx)
            hx, bhx = hx_fn(ghx) if hx_fn is not None else (ghx, ghx)
            return self.critic_linear(hx), self.actor_linear(hx), hx, (ghx, bhx, c_input, input_x, input_tanh)
        else:
            c_input = self.input_ff(input)
            c_input = c_input.view(-1, self.input_c_features)
            input, input_x, input_tanh = input_fn(c_input) if input_fn is not None else (c_input, c_input)
            return c_input, input_x, input_tanh

    def init_hidden(self, batch_size=1):
        return torch.zeros(batch_size, self.gru_units)

    def get_action_linear(self, state):
        return self.actor_linear(state)

    def transact(self, o_x, hx):
        hx = self.gru(o_x, hx)
        return hx


class ControlObsQBNet(nn.Module):
    """
    Quantized Bottleneck Network(QBN) for observation features
    """
    def __init__(self, input_size, x_features):
        super(ControlObsQBNet, self).__init__()
        self.bhx_size = x_features

        f1 = int(8 * x_features)
        self.encoder = nn.Sequential(nn.Linear(input_size, f1),
                                     nn.Tanh(),
                                     nn.Linear(f1, x_features),
                                     TernaryTanh())

        self.decoder = nn.Sequential(nn.Linear(x_features, f1),
                                     nn.Tanh(),
                                     nn.Linear(f1, input_size),
                                     nn.ReLU6())


    def forward(self, x):
        encoded, before_ttanh = self.encode(x)
        # encoded = self.encode(x)
        decoded = self.decode(encoded)
        return decoded, encoded, before_ttanh
        # return decoded, encoded

    def encode(self, x):
        linear1 = self.encoder[0](x)
        tanh = self.encoder[1](linear1)
        linear2 = self.encoder[2](tanh)
        ttanh = self.encoder[3](linear2)
        return self.encoder(x), linear2
        # return self.encoder(x)

    def decode(self, x):
        return self.decoder(x)


class ControlHxQBNet(nn.Module):
    """
    Quantized Bottleneck Network(QBN) for hidden states of GRU
    """

    def __init__(self, input_size, x_features):
        super(ControlHxQBNet, self).__init__()
        self.bhx_size = x_features
        f1 = int(8 * x_features)
        self.encoder = nn.Sequential(nn.Linear(input_size, f1),
                                     nn.Tanh(),
                                     nn.Linear(f1, x_features),
                                     TernaryTanh())

        self.decoder = nn.Sequential(nn.Linear(x_features, f1),
                                     nn.Tanh(),
                                     nn.Linear(f1, input_size),
                                     nn.Tanh())

    def forward(self, x):
        # encoded, before_ttanh = self.encode(x)
        encoded = self.encode(x)
        decoded = self.decode(encoded)
        # return decoded, encoded, before_ttanh
        return decoded, encoded

    def encode(self, x):
        linear1 = self.encoder[0](x)
        tanh = self.encoder[1](linear1)
        linear2 = self.encoder[2](tanh)
        ttanh = self.encoder[3](linear2)
        # return self.encoder(x), linear2
        return self.encoder(x)

    def decode(self, x):
        return self.decoder(x)


class ControlMMNet(nn.Module):
    """
    Moore Machine Network(MMNet) definition
    """
    def __init__(self, net, hx_qbn=None, obs_qbn=None):
        super(ControlMMNet, self).__init__()
        self.bhx_units = hx_qbn.bhx_size if hx_qbn is not None else None
        self.gru_units = net.gru_units
        self.obx_net = obs_qbn
        self.gru_net = net
        self.bhx_net = hx_qbn
        self.actor_linear = self.gru_net.get_action_linear

    def init_hidden(self, batch_size=1):
        return self.gru_net.init_hidden(batch_size)

    def forward(self, x, inspect=False):
        if inspect:
            x, hx = x
            critic, actor, hx, (ghx, bhx, input_c, input_x, input_tanh) = self.gru_net((x, hx), input_fn=self.obx_net,
                                                                           hx_fn=self.bhx_net, inspect=True)
            return critic, actor, hx, (ghx, bhx), (input_c, input_x, input_tanh)
        else:
            input_c = self.gru_net(x, input_fn=self.obx_net, hx_fn=self.bhx_net, inspect=False)
            return input_c

    def get_action_linear(self, state, decode=False):
        if decode:
            hx = self.bhx_net.decode(state)
        else:
            hx = state
        return self.actor_linear(hx)

    def transact(self, o_x, hx_x):
        hx_x = self.gru_net.transact(self.obx_net.decode(o_x), self.bhx_net.decode(hx_x))
        _, hx_x = self.bhx_net(hx_x)
        return hx_x

    def state_encode(self, state):
        return self.bhx_net.encode(state)

    def obs_encode(self, obs, hx=None):
        if hx is None:
            hx = Variable(torch.zeros(1, self.gru_units))
            if next(self.parameters()).is_cuda:
                hx = hx.cuda()
        _, _, _, (_, _, _, input_x, _, _) = self.gru_net((obs, hx), input_fn=self.obx_net, hx_fn=self.bhx_net, inspect=True)
        return input_x