# -*- coding: utf-8 -*-
import numpy as np
from scipy.ndimage import uniform_filter, gaussian_filter, gaussian_gradient_magnitude, gaussian_laplace
import logging

logger = logging.getLogger(__name__)

# To show debug messages, uncomment this
# from outside the module, import the logger object and use something like this
# logger.setLevel(logging.DEBUG)

# add a console handler
ch = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)


def noise_stats(data, tol=1e-2):

    d = data[data > 0].flatten()

    # find the quartiles of the non-zero data
    q1, q2, q3 = np.percentile(d, [25, 50, 75])
    logger.debug('Quartiles of the original data: {}, {}, {}'.format(q1, q2, q3))

    # find the quartiles of the non-zero data that is less than a cutoff
    # start with the first quartile and then iterate using the upper fence
    uf = q1

    # repeat
    for it in range(20):
        q1, q2, q3 = np.percentile(d[d < uf], [25, 50, 75])
        logger.debug('Iteration {}. Quartiles of the trimmed data: {}, {}, {}'.format(it, q1, q2, q3))
        q13 = q3 - q1
        ufk = q2 + 1.5*q13
        # check for convergence
        if abs(ufk-uf)/uf < tol or ufk < tol:
            break
        else:
            uf = ufk
    else:
        logger.warning('Warning, number of iterations exceeded')

    # recompute the quartiles
    q1, q2, q3 = np.percentile(d[d < uf], [25, 50, 75])
    q13 = q3 - q1
    # q1, q2, q3 describes the noise
    # anything above this is a noise outlier above (possibly signal)
    uf = q2 + 1.5*q13
    # anything below lf is a signal outlier below (not useful)
    lf = q2 - 1.5*q13
    # but remember that the noise distribution is NOT symmetric, so uf is an underestimate

    return lf, q2, uf


def signal_likelihood(data):
    """Return a likelihood that data is signal

    in SNR units, sigmoid with width 1, shifted to the right by 1
    ie P(<1)=0, P(2)=0.46, P(3)=0.76, P(4)=0.01, P(5)=0.96
    """
    _, _, uf = noise_stats(data)

    # The probability that each point has a signal
    p = (data > uf) * (-1 + 2 / (1+np.exp(-(data-uf)/uf)))

    return p


def coil_correction(data, box_size=10, auto_scale=False):
    """Weighted least squares estimate of the coil intensity correction

    :param data: 3D or 4D array or list of 3D arrays
    :param box_size: size over which to smooth
    :return: 3D array coil correction
    """
    # <data*w>/<data**2*w>
    # if we're given a list, copy into a 4D array
    if isinstance(data, list):
        nx, ny, nz = data[0].shape
        nc = len(data)
        h = np.zeros([nx, ny, nz, nc])
        for n in range(nc):
            h[:, :, :, n] = data[n]
    else:
        h = data

    if len(h.shape) == 4:
        nc = h.shape[3]
        a = np.zeros(h.shape[0:3])
        b = np.zeros(h.shape[0:3])
        for n in range(nc):
            t = h[:, :, :, n]
            a = a + uniform_filter(t, box_size, mode='constant')
            b = b + uniform_filter(t**2, box_size, mode='constant')
    else:
        a = uniform_filter(h, box_size, mode='constant')
        b = uniform_filter(h**2, box_size, mode='constant')

    mask = signal_likelihood(a) > 0.8
    c = np.zeros(a.shape)
    c[mask] = a[mask] / b[mask]

    # Scale if desired
    if auto_scale:
        if len(h.shape) == 4:
            nc = h.shape[3]
            d = np.zeros(h.shape)
            for n in range(nc):
                d = c * h[:, :, :, n]
        else:
            d = c * h
        scale = np.sum(d.flatten()) / np.sum(h.flatten())
        c = scale * c

    return c


def textures(data, scales=5):
    """Compute image textures at a particular scale or set of scales
    gaussian smoothing, gradient magnitude, laplacian and standard deviation

    :param data:  3D numpy array
    :param scales: int or list of ints
    :return: 4D numpy array
    """
    assert len(data.shape) == 3
    nx, ny, nz = data.shape

    if isinstance(scales, int):
        scales = [scales]
    ns = len(scales)

    t = np.zeros([nx, ny, nz, 4*ns])
    for s in range(ns):
        t[:, :, :, 4*s+0] = gaussian_filter(data, sigma=scales[s], mode='constant')
        t[:, :, :, 4*s+1] = gaussian_gradient_magnitude(data, sigma=scales[s], mode='constant')
        t[:, :, :, 4*s+2] = gaussian_laplace(data, sigma=scales[s], mode='constant')
        t[:, :, :, 4*s+3] = np.sqrt(gaussian_filter((data - t[:, :, :, 0])**2, sigma=scales[s], mode='constant'))

    return t
