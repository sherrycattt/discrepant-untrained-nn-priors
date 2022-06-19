import torch
import torch.nn as nn

from net.layers import act, conv


class DeepDecoder(nn.Module):
    def __init__(
            self,
            num_output_channels=1,
            num_channels_up=[16] * 5,
            filter_size_up=3,
            need_sigmoid=True,
            pad='reflection',
            upsample_mode='bilinear',
            act_fun='ReLU',
            bn_affine=True,
            need1x1_up=False,
            reg_std=0.03,
    ):
        super(DeepDecoder, self).__init__()
        self.activations = []
        self.num_channels_up = num_channels_up
        self.need1x1_up = need1x1_up
        self.reg_std = nn.Parameter(torch.tensor(reg_std), requires_grad=False)
        n_scales = len(num_channels_up)

        if not (isinstance(filter_size_up, list) or isinstance(filter_size_up, tuple)):
            filter_size_up = [filter_size_up] * n_scales
        model = nn.Sequential()

        input_depth = num_channels_up[0]
        for i in range(n_scales):
            model.add(conv(input_depth, num_channels_up[i], filter_size_up[i], stride=1, pad=pad, bias=False))
            model.add(nn.Upsample(scale_factor=2, mode=upsample_mode))
            model.add(act(act_fun))
            model.add(nn.BatchNorm2d(num_channels_up[i], affine=bn_affine))

            if need1x1_up:
                model.add(conv(num_channels_up[i], num_channels_up[i], kernel_size=1, stride=1, pad=pad, bias=False))
                model.add(act(act_fun))
                model.add(nn.BatchNorm2d(num_channels_up[i], affine=bn_affine))

            input_depth = num_channels_up[i]

        model.add(conv(num_channels_up[-1], num_channels_up[-1], kernel_size=filter_size_up[-1], stride=1, pad=pad,
                       bias=False))
        model.add(act(act_fun))
        model.add(nn.BatchNorm2d(num_channels_up[-1], affine=bn_affine))
        model.add(conv(num_channels_up[-1], num_output_channels, kernel_size=1, stride=1, pad=pad, bias=False))
        if need_sigmoid:
            model.add(nn.Sigmoid())
        self.net = model

    def forward(self, *inputs, **kwargs):
        return self.net(*inputs, **kwargs)
