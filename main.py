import argparse
import sys

import numpy as np
import torch
import torch.nn.functional as F
from pathlib2 import Path

try:
    from skimage.metrics import peak_signal_noise_ratio as compare_psnr
except ImportError:
    from skimage.measure import compare_psnr

from net.curve_model import SCurve
from net.decoder_model import DeepDecoder
from net.losses import HistEntropyLoss, FiedelityLoss
from net.noise import get_noise
from net.skip_model import SkipAdaDrop
from utils.image_io import np_to_torch, torch_to_np, save_image, prepare_image
from utils.imresize import np_imresize


def downsample(image):
    return F.avg_pool2d(image, kernel_size=32, stride=16, padding=0)


class Engine(object):
    def __init__(self, input_path, output_dir, device, num_iter=15000, show_every=1000, drop_tau=0.1,
                 drop_mod_every=10000, num_inf_iter=100, input_depth=8, n_scale=5, ):
        print(f"Processing {input_path}")

        self.output_dir = output_dir
        self.num_iter = num_iter
        self.show_every = show_every
        self.drop_mod_every = drop_mod_every
        self.num_inf_iter = num_inf_iter

        self.total_loss = None
        self.learning_rate = 0.001

        # init images
        self.image_name = Path(input_path).stem
        self.image = prepare_image(input_path)
        self.original_image = self.image.copy()
        factor = 1
        while self.image.shape[1] >= 800 or self.image.shape[2] >= 800:
            new_shape_x, new_shape_y = self.image.shape[1] / factor, self.image.shape[2] / factor
            new_shape_x -= (new_shape_x % 32)
            new_shape_y -= (new_shape_y % 32)
            self.image = np_imresize(self.image, output_shape=(new_shape_x, new_shape_y))
            factor += 1
        self.image_torth = np_to_torch(self.image).float().to(device)
        self.illum_ref = downsample(self.image_torth.max(dim=1, keepdim=True)[0]).detach()

        # init nets
        self.illum_net = DeepDecoder(num_output_channels=1, num_channels_up=[16] * n_scale)
        self.illum_net = self.illum_net.to(device)

        self.reflect_net = SkipAdaDrop(
            num_input_channels=input_depth, num_output_channels=3, tau=drop_tau, size=self.image.shape,
            num_channels_down=[128] * n_scale, num_channels_up=[128] * n_scale, num_channels_skip=[4] * n_scale,
        )
        self.reflect_net = self.reflect_net.to(device)

        self.scurve_net = SCurve(size=self.image.shape)
        self.scurve_net = self.scurve_net.to(device)

        # init inputs
        self.reflect_net_inputs = get_noise(
            input_depth, 'noise', (self.image.shape[1], self.image.shape[2]), var=1 / 10.
        ).float().to(device).detach()

        self.illum_net_inputs = torch.zeros(
            [1, 16] + [int(s / (2 ** n_scale)) for s in self.image.shape[-2:]]
        ).float().to(device).uniform_().mul(1. / 10).detach()

        # init parameters
        self.parameters = [p for p in self.reflect_net.parameters()] + \
                          [p for p in self.illum_net.parameters()] + \
                          [p for p in self.scurve_net.parameters()]

        # init loss
        self.recon_criterion = FiedelityLoss().to(device)
        self.hist_criterion = HistEntropyLoss().to(device)

    def optimize(self):
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True

        optimizer = torch.optim.Adam(self.parameters, lr=self.learning_rate)
        psnr = 0.
        for step in range(1, self.num_iter + 1):
            optimizer.zero_grad()

            self.illum_net.activations = []

            illum_net_input = self.illum_net_inputs + self.illum_net_inputs.clone().normal_() * self.illum_net.reg_std.data
            illum_out = self.illum_net(illum_net_input)

            if (step % self.drop_mod_every) == (self.drop_mod_every // 2) and (
                    step + (self.drop_mod_every // 2)) < self.num_iter:
                print(f"\nThe dropout maps are modified at iteration {step}")
                self.reflect_net.modified_keep_prob_map(illum_out)

            reflect_net_input = self.reflect_net_inputs + self.reflect_net_inputs.clone().normal_() * self.reflect_net.reg_std.data
            reflect_out = self.reflect_net(reflect_net_input)

            illum_en = self.scurve_net(illum_out.clamp(min=0., max=1.).detach(), self.image_torth)

            image_en = illum_en * reflect_out.detach()
            image_out = illum_out * reflect_out

            recon_loss = self.recon_criterion(y=self.image_torth, y_pred=image_out)
            bright_loss = F.mse_loss(downsample(illum_out), self.illum_ref)
            hist_loss = self.hist_criterion(image_en)
            self.total_loss = recon_loss + 0.01 * bright_loss + 1e-6 * hist_loss
            self.total_loss.backward(retain_graph=True)

            optimizer.step()

            # plot results and calculate PSNR
            if step % self.show_every == 0:
                image_out_np = np.clip(torch_to_np(image_out.detach()), 0, 1)
                psnr = compare_psnr(self.image, image_out_np)

                image_en_np = np.clip(torch_to_np(image_en.detach()), 0, 1)
                image_en_np = np_imresize(image_en_np, output_shape=self.original_image.shape[1:])
                save_image(f"{self.image_name}_out_{step}", image_en_np, self.output_dir)

            # obtain current result
            if step % 8 == 0:
                print('Iteration: %05d    Loss: %.6f   PSNR: %.2f ' % (step, self.total_loss.item(), psnr), '\r', end='')

    def inference(self):
        with torch.no_grad():
            illum = self.illum_net(self.illum_net_inputs)
            illum_en = self.scurve_net(illum, self.image_torth)

            reflect_avg = None
            for step in range(self.num_inf_iter):
                reflect = self.reflect_net(self.reflect_net_inputs)
                if reflect_avg is None:
                    reflect_avg = reflect.detach()
                else:
                    reflect_avg = (reflect_avg * (step - 1) + reflect.detach()) / step

            image_en_avg = np.clip(torch_to_np((illum_en * reflect_avg).detach()), 0, 1)
            image_en_avg = np_imresize(image_en_avg, output_shape=self.original_image.shape[1:])
            save_image(f"{self.image_name}_out_final", image_en_avg, self.output_dir)

            print(f"Done. Please check the results in {self.output_dir}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='PyTorch implementation of "Discrepant Untrained Network Priors"')
    parser.add_argument('--no_cuda', action='store_true', help='Use cuda?')
    parser.add_argument('--input_path', default='images/input2.png',
                        help='Path to input')
    parser.add_argument('--output_dir', default='output',
                        help='Path to save dir')
    parser.add_argument('--num_iter', type=int, default=15000,
                        help='Total iterations (default: 100000)')
    parser.add_argument('--show_every', type=int, default=1000,
                        help='How often to show the results (default: 1000)')
    parser.add_argument('--drop_tau', type=float, default=0.1,
                        help='Denoising stength for dropout ensemble (default: 0.1)')

    args = parser.parse_args()
    if (not args.no_cuda) and (not torch.cuda.is_available()):
        print("ERROR: cuda is not available, try running on CPU with option --no_cuda")
        sys.exit(1)

    device = torch.device("cuda" if not args.no_cuda else "cpu")
    engine = Engine(input_path=args.input_path, output_dir=args.output_dir, device=device,
                    num_iter=args.num_iter, show_every=args.show_every, drop_tau=args.drop_tau, )
    engine.optimize()
    engine.inference()
