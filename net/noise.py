import numpy as np
import torch

from utils.image_io import np_to_torch


def get_noise(input_depth, method, spatial_size, noise_type='u', var=1. / 100):
    """
    Returns a pytorch.Tensor of size (1 x `input_depth` x `spatial_size[0]` x `spatial_size[1]`)
    initialized in a specific way.
    Args:
        input_depth: number of channels in the tensor
        method: `noise` for fillting tensor with noise; `meshgrid` for np.meshgrid
        spatial_size: spatial size of the tensor to initialize
        noise_type: 'u' for uniform; 'n' for normal
        var: a factor, a noise will be multiplicated by. Basically it is standard deviation scaler.
    """
    if isinstance(spatial_size, int):
        spatial_size = (spatial_size, spatial_size)
    if method == 'noise':
        shape = [1, input_depth, spatial_size[0], spatial_size[1]]
        net_input = torch.zeros(shape)

        fill_noise(net_input, noise_type)
        net_input *= var
    elif method == 'meshgrid':
        assert input_depth % 2 == 0
        X, Y = np.meshgrid(np.arange(0, spatial_size[1]) / float(spatial_size[1] - 1),
                           np.arange(0, spatial_size[0]) / float(spatial_size[0] - 1))
        meshgrid = np.concatenate([X[None, :], Y[None, :]] * (input_depth // 2))
        net_input = np_to_torch(meshgrid)
    else:
        assert False

    return net_input


def fill_noise(x, noise_type):
    """
    Fills tensor `x` with noise of type `noise_type`.
    """
    if noise_type == 'u':
        x.uniform_(-0.5, 0.5)
    elif noise_type == 'n':
        x.normal_()
    else:
        assert False
