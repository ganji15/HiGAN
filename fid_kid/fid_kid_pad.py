#!/usr/bin/env python3
"""Calculates the Frechet Inception Distance (FID) and The Kernel Inception Distance (KID) to evalulate GANs
The FID and KID is calculated by assuming that X_1 and X_2 are the activations of
the pool_3 layer of the inception net for generated samples and real world
samples respectively.
Code apapted from https://github.com/bioinf-jku/TTUR and  https://github.com/mbinkowski/MMD-GAN
"""

import numpy as np
import torch
from scipy import linalg
import cv2
from torch.nn.functional import adaptive_avg_pool2d
from sklearn.metrics.pairwise import polynomial_kernel
# import matplotlib.pyplot as plt
import sys

try:
    from tqdm import tqdm
except ImportError:
    # If not tqdm is not available, provide a mock version of it
    def tqdm(x): return x

from fid_kid.inception import InceptionV3
# from inception import InceptionV3


def get_activations(data_source, n_batches, max_img_width, model, dims, device):
    """Calculates the activations of the pool_3 layer for all images.
    Params:
    -- data_source  : List of image files paths
    -- model       : Instance of inception model
    -- batch_size  : Batch size of images for the model to process at once.
                     Make sure that the number of samples is a multiple of
                     the batch size, otherwise some samples are ignored. This
                     behavior is retained to match the original FID score
                     implementation.
    -- dims        : Dimensionality of features returned by Inception
    -- cuda        : If set to True, use GPU
    -- verbose     : If set to True and parameter out_step is given, the number
                     of calculated batches is reported.
    Returns:
    -- A numpy array of dimension (num images, dims) that contains the
       activations of the given tensor when feeding inception with the
       query tensor.
    """
    model.eval()
    pred_arr = []
    for batch in tqdm(data_source, total=n_batches):
        imgs, img_lens = batch[:2]
        imgs, img_lens = imgs.to(device), img_lens.to(device)
        imgs = (imgs + 1) / 2
        # print('imgs min:{} max:{}'.format(imgs.min().item(), imgs.max().item()))

        if imgs.size(1) == 1:
            imgs = imgs.repeat(1, 3, 1, 1)

        if imgs.size(-1) < max_img_width:
            pad_width = max_img_width - imgs.size(-1)
            imgs = torch.nn.functional.pad(imgs, [0, pad_width, 0, 0], value=0.)
            # import matplotlib.pyplot as plt
            # plt.subplot(211)
            # plt.imshow(imgs[0, 0].cpu().numpy(), cmap='binary')
            # plt.subplot(212)
            # plt.imshow(imgs[1, 0].cpu().numpy(), cmap='binary')
            # plt.show()

        with torch.no_grad():
            # print(images.type())
            pred = model(imgs)[0]

        # If model output is not scalar, apply global spatial average pooling.
        # This happens if you choose a dimensionality not equal 2048.
        if pred.size(2) != 1 or pred.size(3) != 1:
            pred = adaptive_avg_pool2d(pred, output_size=(1, 1))

        # pred_arr[start:end] = pred.cpu().data.numpy().reshape(pred.size(0), -1)
        pred_arr.append(pred.cpu().data.numpy().reshape(pred.size(0), -1))

    pred_arr = np.concatenate(pred_arr, axis=0)
    assert pred_arr.shape[-1] == dims
    return pred_arr


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
    Stable version by Dougal J. Sutherland.
    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.
    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) +
            np.trace(sigma2) - 2 * tr_covmean)


def calculate_activation_statistics(*args, **kwargs):
    """Calculation of the statistics used by the FID.
    Returns:
    -- mu    : The mean over samples of the activations of the pool_3 layer of
               the inception model.
    -- sigma : The covariance matrix of the activations of the pool_3 layer of
               the inception model.
    """
    act = get_activations(*args, **kwargs)
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return act, mu, sigma


def polynomial_mmd_averages(codes_g, codes_r, n_subsets=50, subset_size=1000,
                            ret_var=True, output=sys.stdout, **kernel_args):
    m = min(codes_g.shape[0], codes_r.shape[0])
    mmds = np.zeros(n_subsets)
    if ret_var:
        vars = np.zeros(n_subsets)
    choice = np.random.choice

    if subset_size > len(codes_g):
        print(('Warning: subset_size is bigger than len(codes_g). [sub:{}  code_g:{}]'.format(subset_size, len(codes_g))))
        subset_size = len(codes_g)
    if subset_size > len(codes_r):
        print(('Warning: subset_size is bigger than len(codes_r). [sub:{}  code_g:{}]'.format(subset_size, len(codes_r))))
        subset_size = len(codes_r)

    with tqdm(range(n_subsets), desc='MMD', file=output) as bar:
        for i in bar:
            g = codes_g[choice(len(codes_g), subset_size, replace=False)]
            r = codes_r[choice(len(codes_r), subset_size, replace=False)]
            # print(subset_size)
            # print(g.shape)
            # print(r.shape)
            o = polynomial_mmd(g, r, **kernel_args, var_at_m=m, ret_var=ret_var)
            if ret_var:
                mmds[i], vars[i] = o
            else:
                mmds[i] = o
            bar.set_postfix({'mean': mmds[:i+1].mean()})
    return (mmds, vars) if ret_var else mmds


def polynomial_mmd(codes_g, codes_r, degree=3, gamma=None, coef0=1,
                   var_at_m=None, ret_var=True):
    # use  k(x, y) = (gamma <x, y> + coef0)^degree
    # default gamma is 1 / dim
    X = codes_g
    Y = codes_r

    K_XX = polynomial_kernel(X, degree=degree, gamma=gamma, coef0=coef0)
    K_YY = polynomial_kernel(Y, degree=degree, gamma=gamma, coef0=coef0)
    K_XY = polynomial_kernel(X, Y, degree=degree, gamma=gamma, coef0=coef0)

    return _mmd2_and_variance(K_XX, K_XY, K_YY,
                              var_at_m=var_at_m, ret_var=ret_var)


def _sqn(arr):
    flat = np.ravel(arr)
    return flat.dot(flat)


def _mmd2_and_variance(K_XX, K_XY, K_YY, unit_diagonal=False,
                       mmd_est='unbiased', block_size=1024,
                       var_at_m=None, ret_var=True):
    # based on
    # https://github.com/dougalsutherland/opt-mmd/blob/master/two_sample/mmd.py
    # but changed to not compute the full kernel matrix at once
    m = K_XX.shape[0]
    assert K_XX.shape == (m, m)
    assert K_XY.shape == (m, m)
    assert K_YY.shape == (m, m)
    if var_at_m is None:
        var_at_m = m

    # Get the various sums of kernels that we'll use
    # Kts drop the diagonal, but we don't need to compute them explicitly
    if unit_diagonal:
        diag_X = diag_Y = 1
        sum_diag_X = sum_diag_Y = m
        sum_diag2_X = sum_diag2_Y = m
    else:
        diag_X = np.diagonal(K_XX)
        diag_Y = np.diagonal(K_YY)

        sum_diag_X = diag_X.sum()
        sum_diag_Y = diag_Y.sum()

        sum_diag2_X = _sqn(diag_X)
        sum_diag2_Y = _sqn(diag_Y)

    Kt_XX_sums = K_XX.sum(axis=1) - diag_X
    Kt_YY_sums = K_YY.sum(axis=1) - diag_Y
    K_XY_sums_0 = K_XY.sum(axis=0)
    K_XY_sums_1 = K_XY.sum(axis=1)

    Kt_XX_sum = Kt_XX_sums.sum()
    Kt_YY_sum = Kt_YY_sums.sum()
    K_XY_sum = K_XY_sums_0.sum()

    if mmd_est == 'biased':
        mmd2 = ((Kt_XX_sum + sum_diag_X) / (m * m)
                + (Kt_YY_sum + sum_diag_Y) / (m * m)
                - 2 * K_XY_sum / (m * m))
    else:
        assert mmd_est in {'unbiased', 'u-statistic'}
        mmd2 = (Kt_XX_sum + Kt_YY_sum) / (m * (m-1))
        if mmd_est == 'unbiased':
            mmd2 -= 2 * K_XY_sum / (m * m)
        else:
            mmd2 -= 2 * (K_XY_sum - np.trace(K_XY)) / (m * (m-1))

    if not ret_var:
        return mmd2

    Kt_XX_2_sum = _sqn(K_XX) - sum_diag2_X
    Kt_YY_2_sum = _sqn(K_YY) - sum_diag2_Y
    K_XY_2_sum = _sqn(K_XY)

    dot_XX_XY = Kt_XX_sums.dot(K_XY_sums_1)
    dot_YY_YX = Kt_YY_sums.dot(K_XY_sums_0)

    m1 = m - 1
    m2 = m - 2
    zeta1_est = (
        1 / (m * m1 * m2) * (
            _sqn(Kt_XX_sums) - Kt_XX_2_sum + _sqn(Kt_YY_sums) - Kt_YY_2_sum)
        - 1 / (m * m1)**2 * (Kt_XX_sum**2 + Kt_YY_sum**2)
        + 1 / (m * m * m1) * (
            _sqn(K_XY_sums_1) + _sqn(K_XY_sums_0) - 2 * K_XY_2_sum)
        - 2 / m**4 * K_XY_sum**2
        - 2 / (m * m * m1) * (dot_XX_XY + dot_YY_YX)
        + 2 / (m**3 * m1) * (Kt_XX_sum + Kt_YY_sum) * K_XY_sum
    )
    zeta2_est = (
        1 / (m * m1) * (Kt_XX_2_sum + Kt_YY_2_sum)
        - 1 / (m * m1)**2 * (Kt_XX_sum**2 + Kt_YY_sum**2)
        + 2 / (m * m) * K_XY_2_sum
        - 2 / m**4 * K_XY_sum**2
        - 4 / (m * m * m1) * (dot_XX_XY + dot_YY_YX)
        + 4 / (m**3 * m1) * (Kt_XX_sum + Kt_YY_sum) * K_XY_sum
    )
    var_est = (4 * (var_at_m - 2) / (var_at_m * (var_at_m - 1)) * zeta1_est
               + 2 / (var_at_m * (var_at_m - 1)) * zeta2_est)

    return mmd2, var_est


def calculate_kid_fid(cfg, data_loader, generator, max_img_width, device):
    '''
    ATTENTION: the backgroud value of input images must be -1, and the foreground values should be less than 1.
    '''
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[cfg.dims]

    inceptionV3_model = InceptionV3([block_idx])
    inceptionV3_model.to(device)

    n_batches = len(data_loader)
    act1, m1, s1 = calculate_activation_statistics(data_loader, n_batches, max_img_width, inceptionV3_model, cfg.dims, device)
    act2, m2, s2 = calculate_activation_statistics(generator, n_batches, max_img_width, inceptionV3_model, cfg.dims, device)
    fid_value = calculate_frechet_distance(m1, s1, m2, s2)

    ret = polynomial_mmd_averages(
            act1, act2, degree=cfg.mmd_degree, gamma=cfg.mmd_gamma,
            coef0=cfg.mmd_coef0, ret_var=cfg.mmd_var,
            n_subsets=cfg.mmd_subsets, subset_size=cfg.mmd_subset_size)
    if cfg.mmd_var:
        mmd2s, vars = ret
    else:
        mmd2s = ret

    return {'FID': fid_value, 'KID': mmd2s.mean()*100}

