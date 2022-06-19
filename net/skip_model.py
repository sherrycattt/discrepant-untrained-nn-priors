import torch
import torch.nn as nn
import torch.nn.functional as F
from kornia.filters import GaussianBlur2d

from net.layers import Concat, bn, conv, act, AdaptiveDropout


class SkipAdaDrop(nn.Module):
    """
    Assembles encoder-decoder with skip connections.

    Arguments:
        act_fun: Either string 'LeakyReLU|ReLU|ELU|none' or module (e.g. nn.ReLU)
        pad (string): zero|reflection (default: 'zero')
        upsample_mode (string): 'nearest|bilinear' (default: 'nearest')
    """

    def __init__(
            self,
            num_input_channels=2,
            num_output_channels=3,
            num_channels_down=[16, 32, 64, 128, 128],
            num_channels_up=[16, 32, 64, 128, 128],
            num_channels_skip=[4, 4, 4, 4, 4],
            filter_size_down=3,
            filter_size_up=3,
            filter_skip_size=1,
            need_sigmoid=True,
            need_bias=True,
            pad='reflection',
            upsample_mode='bilinear',
            act_fun='LeakyReLU',
            need1x1_up=True,
            tau=0.05,
            size=[512, 512],
            reg_std=0.03,
    ):
        super(SkipAdaDrop, self).__init__()
        assert len(num_channels_down) == len(num_channels_up) == len(num_channels_skip)

        self.reg_std = nn.Parameter(torch.tensor(reg_std), requires_grad=False)
        self.num_channels_down = num_channels_down
        self.num_channels_up = num_channels_up
        self.num_channels_skip = num_channels_skip
        self.filter_size_down = filter_size_down
        self.filter_size_up = filter_size_up
        self.upsample_mode = upsample_mode
        self.tau = tau

        self.blur = GaussianBlur2d(kernel_size=(3, 3), sigma=(2. / 3, 2. / 3), border_type='reflect')
        self.activations = []

        n_scales = len(num_channels_down)

        if not (isinstance(upsample_mode, list) or isinstance(upsample_mode, tuple)):
            upsample_mode = [upsample_mode] * n_scales

        if not (isinstance(filter_size_down, list) or isinstance(filter_size_down, tuple)):
            filter_size_down = [filter_size_down] * n_scales

        if not (isinstance(filter_size_up, list) or isinstance(filter_size_up, tuple)):
            filter_size_up = [filter_size_up] * n_scales

        last_scale = n_scales - 1

        model = nn.Sequential()
        model_tmp = model

        up_size = [1, 1, ] + [s for s in size[-2:]]
        down_size = [1, 1, ] + [s for s in size[-2:]]
        input_depth = num_input_channels
        for i in range(len(num_channels_down)):

            deeper = nn.Sequential()
            skip = nn.Sequential()

            if num_channels_skip[i] != 0:
                model_tmp.add(Concat(1, skip, deeper))
            else:
                model_tmp.add(deeper)

            model_tmp.add(
                bn(num_channels_skip[i] + (num_channels_up[i + 1] if i < last_scale else num_channels_down[i])))

            if num_channels_skip[i] != 0:
                skip.add(conv(input_depth, num_channels_skip[i], filter_skip_size, bias=need_bias, pad=pad))
                skip.add(bn(num_channels_skip[i]))
                skip.add(act(act_fun))

            deeper.add(conv(input_depth, num_channels_down[i], filter_size_down[i], 2, bias=need_bias, pad=pad))
            deeper.add(bn(num_channels_down[i]))
            deeper.add(act(act_fun))

            down_size = [1, 1, ] + [s // 2 for s in down_size[-2:]]
            deeper.add(AdaptiveDropout(size=down_size, tau=self.tau))

            deeper.add(conv(num_channels_down[i], num_channels_down[i], filter_size_down[i], bias=need_bias, pad=pad))
            deeper.add(bn(num_channels_down[i]))
            deeper.add(act(act_fun))

            deeper_main = nn.Sequential()

            if i == len(num_channels_down) - 1:
                # The deepest
                k = num_channels_down[i]
            else:
                deeper.add(deeper_main)
                k = num_channels_up[i + 1]

            deeper.add(nn.Upsample(scale_factor=2, mode=upsample_mode[i], align_corners=True))

            model_tmp.add(
                conv(num_channels_skip[i] + k, num_channels_up[i], filter_size_up[i], 1, bias=need_bias, pad=pad))
            model_tmp.add(bn(num_channels_up[i]))
            model_tmp.add(act(act_fun))

            model_tmp.add(AdaptiveDropout(size=up_size, tau=self.tau))
            up_size = [1, 1, ] + [s // 2 for s in up_size[-2:]]

            if need1x1_up:
                model_tmp.add(conv(num_channels_up[i], num_channels_up[i], 1, bias=need_bias, pad=pad))
                model_tmp.add(bn(num_channels_up[i]))
                model_tmp.add(act(act_fun))

            input_depth = num_channels_down[i]
            model_tmp = deeper_main

        model.add(conv(num_channels_up[0], num_output_channels, 1, bias=need_bias, pad=pad))
        if need_sigmoid:
            model.add(nn.Sigmoid())
        self.net = model

    def forward(self, *inputs, **kwargs):
        return self.net(*inputs, **kwargs)

    def gen_illum_map_list(self, illum_map):
        illum_map = illum_map.clone().detach()
        illum_map_list = []
        for i in range(len(self.num_channels_up)):
            illum_map = self.blur(illum_map)
            illum_map_list.append(illum_map)
            illum_map = F.interpolate(illum_map, scale_factor=0.5, mode='area', recompute_scale_factor=True)
        for i in range(len(self.num_channels_down), 0, -1):
            illum_map = self.blur(illum_map)
            illum_map_list.append(illum_map)
            illum_map = F.interpolate(illum_map, scale_factor=2, mode='area', recompute_scale_factor=True)
            illum_map = self.blur(illum_map)
        return illum_map_list

    def modified_keep_prob_map(self, illum_map):
        illum_map_list = self.gen_illum_map_list(illum_map)
        illum_map_list.reverse()

        def modified_dropout(m):
            """ This is used to initialize weights of any network """
            class_name = m.__class__.__name__
            if class_name.find('AdaptiveDropout') != -1:
                m.set_keep_prob_map(illum_map_list.pop(0))

        self.net.apply(modified_dropout)
        assert len(illum_map_list) == 0, "the length of illum_map_list is {}".format(len(illum_map_list))
