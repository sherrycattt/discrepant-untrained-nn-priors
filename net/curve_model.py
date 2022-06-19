import torch
import torch.nn as nn

from net.layers import conv


class SCurve(nn.Module):
    def __init__(self, size=[512, 512], num_feature=8, kernel_size=3):
        super(SCurve, self).__init__()
        down_size = [s // (2 ** 5) for s in size[-2:]]
        self.model = nn.Sequential(
            conv(in_f=4, out_f=num_feature, kernel_size=kernel_size), nn.MaxPool2d(2, 2), nn.LeakyReLU(),
            conv(in_f=num_feature, out_f=num_feature, kernel_size=kernel_size), nn.MaxPool2d(2, 2), nn.LeakyReLU(),
            conv(in_f=num_feature, out_f=num_feature, kernel_size=kernel_size), nn.MaxPool2d(2, 2), nn.LeakyReLU(),
            conv(in_f=num_feature, out_f=num_feature // 2, kernel_size=kernel_size), nn.MaxPool2d(2, 2), nn.LeakyReLU(),
            conv(in_f=num_feature // 2, out_f=1, kernel_size=kernel_size), nn.MaxPool2d(2, 2), nn.LeakyReLU(),
            nn.Flatten(),
            nn.Linear(down_size[0] * down_size[1], 8), nn.ReLU(),
            nn.Linear(8, 1),
        )

    def forward(self, illum, image):
        out = self.model(torch.cat([illum, image], dim=1))
        enhance = torch.exp(torch.log(illum.clamp_min(1e-3)) * torch.sigmoid(out[0][0]))
        return enhance
