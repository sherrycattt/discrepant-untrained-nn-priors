import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftHistogram(nn.Module):
    def __init__(self, bins=255, min=0.0, max=1.0, sigma=30 * 255):
        super(SoftHistogram, self).__init__()
        self.bins = bins
        self.min = min
        self.max = max
        self.sigma = sigma
        self.delta = float(max - min) / float(bins)
        self.centers = nn.Parameter(float(min) + self.delta * (torch.arange(bins).float() + 0.5), requires_grad=False)

    def forward(self, x):
        x = torch.unsqueeze(x, 0) - torch.unsqueeze(self.centers, 1)
        x = torch.sigmoid(self.sigma * (x + self.delta / 2)) - torch.sigmoid(self.sigma * (x - self.delta / 2))
        x = x.sum(dim=1)
        return x


class HistEntropyLoss(nn.Module):
    def __init__(self, bins=256, min=0.0, max=1.0, sigma=30 * 256):
        super(HistEntropyLoss, self).__init__()
        self.softhist = SoftHistogram(bins=bins, min=min, max=max, sigma=sigma)

    def forward(self, x):
        x = torch.exp(torch.mean(torch.log(x.clamp_min(1e-6)), dim=1))
        x = x.view(-1)
        p = self.softhist(x)
        p = p / x.shape[0]
        return 8 + p.mul(p.clamp_min(1e-6).log2()).sum()


class FiedelityLoss(nn.Module):
    def __init__(self):
        super(FiedelityLoss, self).__init__()

    def forward(self, y, y_pred):
        loss = (F.mse_loss(y_pred, y, reduction='none') / (y_pred.abs() + y.abs()).clamp(min=1e-2))
        return loss.mean()
