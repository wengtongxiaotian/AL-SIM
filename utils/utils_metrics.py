import warnings
import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Sequence
import pdb

""" Jax version of https://github.com/VainF/pytorch-msssim """ 

def _fspecial_gauss_1d(size, sigma):
    r"""Create 1-D gauss kernel
    Args:
        size (int): the size of gauss kernel
        sigma (float): sigma of normal distribution
    Returns:
        torch.Tensor: 1D kernel (1 x 1 x size)
    """
    coords = jnp.arange(size, dtype=jnp.float32)
    coords -= size // 2

    g = jnp.exp(-(coords ** 2) / (2 * sigma ** 2))
    g /= g.sum()

    return g[jnp.newaxis, jnp.newaxis, ...]


def gaussian_filter(input, win):
    """ Blur input with 1-D kernel
    Args:
        input: a batch of tensors to be blurred
        window: 1-D gauss kernel
    Returns:
        blurred tensors
    """
    assert all([ws == 1 for ws in win.shape[1:-1]]), win.shape
    assert len(input.shape) == 4

    C = input.shape[1]
    out = input
    for i, s in enumerate(input.shape[2:]):
        if s >= win.shape[-1]:
            out = jax.lax.conv_general_dilated(out, win.swapaxes(2 + i, -1), window_strides =(1, 1), padding='VALID', feature_group_count=C)
        else:
            warnings.warn(
                f"Skipping Gaussian Smoothing at dimension 2+{i} for input: {input.shape} and win size: {win.shape[-1]}"
            )

    return out


def _ssim(X, Y, data_range, win, size_average=True, K=(0.01, 0.03)):

    """ Calculate ssim index for X and Y
    Args:
        X : images
        Y : images
        win : 1-D gauss kernel
        data_range (float or int, optional): value range of input images. (usually 1.0 or 255)
        size_average (bool, optional): if size_average=True, ssim of all images will be averaged as a scalar
    Returns:
        ssim results.
    """
    K1, K2 = K
    # batch, channel, [depth,] height, width = X.shape
    compensation = 1.0

    C1 = (K1 * data_range) ** 2
    C2 = (K2 * data_range) ** 2

    mu1 = gaussian_filter(X, win)
    mu2 = gaussian_filter(Y, win)

    mu1_sq = jnp.square(mu1)
    mu2_sq = jnp.square(mu2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = compensation * (gaussian_filter(X * X, win) - mu1_sq)
    sigma2_sq = compensation * (gaussian_filter(Y * Y, win) - mu2_sq)
    sigma12 = compensation * (gaussian_filter(X * Y, win) - mu1_mu2)

    cs_map = (2 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)  # set alpha=beta=gamma=1
    ssim_map = ((2 * mu1_mu2 + C1) / (mu1_sq + mu2_sq + C1)) * cs_map

    ssim_per_channel = ssim_map.mean((2, 3))
    cs = cs_map.mean((2, 3))
    return ssim_per_channel, cs




def ms_ssim(
    X, Y, data_range=1.0, size_average=True, win_size=11, win_sigma=1.5, win=None, weights=None, K=(0.01, 0.03)
):

    """ interface of ms-ssim
    Args:
        X : a batch of images, (N,C,H,W)
        Y : a batch of images, (N,C,H,W)
        data_range (float or int, optional): value range of input images. (usually 1.0 or 255)
        size_average (bool, optional): if size_average=True, ssim of all images will be averaged as a scalar
        win_size: (int, optional): the size of gauss kernel
        win_sigma: (float, optional): sigma of normal distribution
        win (torch.Tensor, optional): 1-D gauss kernel. if None, a new kernel will be created according to win_size and win_sigma
        weights (list, optional): weights for different levels
        K (list or tuple, optional): scalar constants (K1, K2). Try a larger K2 constant (e.g. 0.4) if you get a negative or NaN results.
    Returns:
        ms-ssim results
    """
    if not X.shape == Y.shape:
        raise ValueError(f"Input images should have the same dimensions, but got {X.shape} and {Y.shape}.")

    # for d in range(len(X.shape) - 1, 1, -1):
    #     pdb.set_trace()
    #     X = jnp.squeeze(X, axis=d)
    #     Y = jnp.squeeze(Y, axis=d)


    if len(X.shape) == 4:
        avg_pool = nn.avg_pool
    else:
        raise ValueError(f"Input images should be 4-d  tensors, but got {X.shape}")

    if win is not None:  # set win_size
        win_size = win.shape[-1]

    if not (win_size % 2 == 1):
        raise ValueError("Window size should be odd.")

    smaller_side = min(X.shape[-2:])
    assert smaller_side > (win_size - 1) * (
        2 ** 4
    ), "Image size should be larger than %d due to the 4 downsamplings in ms-ssim" % ((win_size - 1) * (2 ** 4))

    if weights is None:
        weights = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]
    weights = jnp.array(weights)

    if win is None:
        win = _fspecial_gauss_1d(win_size, win_sigma)
        win = win[jnp.newaxis, ...]
        win = jnp.repeat(win, X.shape[1], axis=0)

    levels = weights.shape[0]
    mcs = []
    for i in range(levels):
        ssim_per_channel, cs = _ssim(X, Y, win=win, data_range=data_range, size_average=False, K=K)

        if i < levels - 1:
            mcs.append(nn.relu(cs))
            padding = [(1,1) for _ in X.shape[2:]]
            X = X.swapaxes(1, -1)
            Y = Y.swapaxes(1, -1)
            X = avg_pool(X, window_shape=(2,2), strides=(2,2), padding=padding)
            Y = avg_pool(Y, window_shape=(2,2), strides=(2,2), padding=padding)
            X = X.swapaxes(1, -1)
            Y = Y.swapaxes(1, -1)

    ssim_per_channel = nn.relu(ssim_per_channel)  # (batch, channel)
    mcs_and_ssim = jnp.stack(mcs + [ssim_per_channel], axis=0)  # (level, batch, channel)
    ms_ssim_val = jnp.prod(mcs_and_ssim ** weights.reshape(-1, 1, 1), axis=0)
    
    if size_average:
        return ms_ssim_val.mean()
    else:
        return ms_ssim_val.mean(1)
