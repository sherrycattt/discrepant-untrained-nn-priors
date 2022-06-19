import numpy as np
import torch
import torch.nn as nn


class Concat(nn.Module):
    def __init__(self, dim, *args):
        super(Concat, self).__init__()
        self.dim = dim

        for idx, module_ in enumerate(args):
            self.add_module(str(idx), module_)

    def forward(self, input_):
        inputs = []
        for module_ in self._modules.values():
            inputs.append(module_(input_))

        inputs_shapes2 = [x.shape[2] for x in inputs]
        inputs_shapes3 = [x.shape[3] for x in inputs]

        if np.all(np.array(inputs_shapes2) == min(inputs_shapes2)) and np.all(
                np.array(inputs_shapes3) == min(inputs_shapes3)):
            inputs_ = inputs
        else:
            target_shape2 = min(inputs_shapes2)
            target_shape3 = min(inputs_shapes3)

            inputs_ = []
            for inp in inputs:
                diff2 = (inp.size(2) - target_shape2) // 2
                diff3 = (inp.size(3) - target_shape3) // 2
                inputs_.append(inp[:, :, diff2: diff2 + target_shape2, diff3:diff3 + target_shape3])

        return torch.cat(inputs_, dim=self.dim)

    def __len__(self):
        return len(self._modules)


def act(act_fun='LeakyReLU'):
    """
        Either string defining an activation function or module (e.g. nn.ReLU)
    """
    if isinstance(act_fun, str):
        if act_fun == 'LeakyReLU':
            return nn.LeakyReLU(0.2, inplace=True)
        elif act_fun == 'ReLU':
            return nn.ReLU()
        elif act_fun == 'ELU':
            return nn.ELU()
        elif act_fun == 'none':
            return nn.Sequential()
        else:
            assert False
    else:
        return act_fun()


def bn(num_features):
    return nn.BatchNorm2d(num_features)


def conv(in_f, out_f, kernel_size, stride=1, bias=True, pad='zero'):
    padder = None
    to_pad = int((kernel_size - 1) / 2)
    if pad == 'reflection':
        padder = nn.ReflectionPad2d(to_pad)
        to_pad = 0

    convolver = nn.Conv2d(in_f, out_f, kernel_size, stride, padding=to_pad, bias=bias)

    layers = [x for x in [padder, convolver] if x is not None]
    return nn.Sequential(*layers)


class AdaptiveDropout(nn.Module):
    def __init__(self, size, tau=0.02):
        super(AdaptiveDropout, self).__init__()
        self.tau = tau
        self.probs = nn.Parameter(torch.ones(size=size), requires_grad=False)
        self.eval()

    def set_keep_prob_map(self, logits):
        probs = torch.sigmoid(logits / self.tau)
        ones = torch.ones_like(probs)
        self.probs.data.copy_(torch.where(probs > 0.99, ones, probs))
        self.train()

    def forward(self, x):
        if self.training:
            noises = torch.bernoulli(self.probs.data.expand_as(x)).float()
            return x.mul(noises.div(self.probs.data))
        else:
            return x


def add_module(self, module_):
    self.add_module(str(len(self) + 1), module_)


torch.nn.Module.add = add_module
