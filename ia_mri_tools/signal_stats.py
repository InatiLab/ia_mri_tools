# -*- coding: utf-8 -*-
import numpy as np
from scipy.ndimage import gaussian_filter

from .logging import logger
from . import FILTER_WIDTH, SIGNAL_SCALE


def noise_stats(data, tol=1e-2):
    """Estimate the statistics of the noise in an image

    Returns the lower fence, median, and upper fence of the trimmed data.
    The upper fence is a lower bound on a (global) estimate of the noise.

    The algorithm alternates between trimming the data and estimating the quartiles.
    At each step, the data are trimmed by selecting the voxels with intensity greater than zero and
    less than the upper fence from the previous step.
    Iteration stops when the relative change in the UF is less than the specified tolerance.
    """

    d = data[data > 0].flatten()

    # find the quartiles of the non-zero data
    q1, q2, q3 = np.percentile(d, [25, 50, 75])
    logger.debug('Quartiles of the original data: {}, {}, {}'.format(q1, q2, q3))

    # find the quartiles of the non-zero data that is less than a cutoff
    # start with the first quartile and then iterate using the upper fence
    uf = q1

    # repeat
    for it in range(20):
        q1, q2, q3 = np.percentile(d[d <= uf], [25, 50, 75])
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
    q1, q2, q3 = np.percentile(d[d <= uf], [25, 50, 75])
    q13 = q3 - q1
    # q1, q2, q3 describes the noise
    # anything above this is a noise outlier above (possibly signal)
    uf = q2 + 1.5*q13
    # anything below lf is a signal outlier below (not useful)
    lf = q2 - 1.5*q13
    # but remember that the noise distribution is NOT symmetric, so uf is an underestimate

    return lf, q2, uf


def signal_stats(data):
    """Estimate the statistics of the signal in an image

    Estimates the noise level and the mean, standard deviation, median, upper and lower fences of the signal
    """

    # Trim the noise
    _, _, uf = noise_stats(data)
    d = data[data > uf].flatten()

    # compute the mean and standard deviation
    m = np.mean(d)
    s = np.std(d)

    # find the quartiles of the signal
    q1, q2, q3 = np.percentile(d, [25, 50, 75])
    logger.debug('Quartiles of the signal: {}, {}, {}'.format(q1, q2, q3))
    q13 = q3 - q1
    # q1, q2, q3 describe the signal, q2 is the median,
    # anything above uf and below lf are signal outliers
    uf = q2 + 1.5*q13
    lf = q2 - 1.5*q13

    return m, s, q2, lf, uf


def signal_likelihood(data, uf=None):
    """Return a likelihood that data is signal

    in SNR units, sigmoid with width 1, shifted to the right by 1
    ie P(<1)=0, P(2)=0.46, P(3)=0.76, P(4)=0.91, P(5)=0.96
    """
    if not uf:
        _, _, uf = noise_stats(data)

    # The probability that each point has a signal
    p = (data > uf) * (-1 + 2 / (1+np.exp(-(data-uf)/uf)))

    return p


def auto_scale(data, signal_min=0, signal_max=2*SIGNAL_SCALE):
    """
    Apply a global brightness and contrast adjustment to an image.

    Uses the signal statistics to clip the data between the lower and upper fences and
    shift and scale to a given range.

    :param data: 2D or 3D image
    :param signal_min: the minimum value of the output image
    :param signal_max: the maximum value of the output image
    :return: clipped, shifted and scaled data
    """

    _, _, _, lf, uf = signal_stats(data)

    # Initialize the output (copy)
    output = data.astype(np.float32)

    # Clip
    output[output < lf] = lf
    output[output > uf] = uf

    # shift and scale
    output = (signal_max - signal_min)/(uf-lf) * (output - lf) + signal_min

    return output


def normalize_global(data):
    """
    Global normalization (z-score) for an image

    :param data: 2D or 3D image
    :return: z-scored data
    """

    m, s, _, _, _ = signal_stats(data)

    # Initialize the output (copy)
    output = data.astype(np.float32)

    # Shift and scale
    output = (output - m) / s

    return output


def normalize_local_2d(data, filter_width=FILTER_WIDTH):
    """
    Local normalization for a feature

    :param data: 2D image or 3D feature image
    :param filter_width: kernel size
    :return: local z-scored image
    """

    assert(data.ndim == 2 or data.ndim == 3)

    # Initialize the output (copy)
    output = data.astype(np.float32)

    # Remove the local mean
    # The 3rd dimension is the number of features
    if data.ndim == 3:
        for n in range(data.shape[2]):
            output[:, :, n] = output[:, :, n] - gaussian_filter(output[:, :, n], filter_width)
        # Compute the standard deviation locally across features and average
        stddev = np.sqrt(gaussian_filter(np.sum(output**2, axis=2), filter_width))
        # Estimate signal floor
        _, _, uf = noise_stats(stddev)
        # Robust division
        for n in range(data.shape[2]):
            output[:, :, n] = output[:, :, n] * stddev / (stddev**2 + uf**2)
    else:
        output = output - gaussian_filter(output, filter_width)
        # Compute the standard deviation locally
        stddev = np.sqrt(gaussian_filter(output**2, filter_width))
        # Estimate signal floor
        _, _, uf = noise_stats(stddev)
        # Robust division
        output = output * stddev / (stddev**2 + uf**2)

    return output


def normalize_local_3d(data, filter_width=FILTER_WIDTH):
    """
    Local normalization for a feature

    :param data: 3D image or 4D feature image
    :param filter_width: kernel size
    :return: local z-scored image
    """

    assert(data.ndim == 3 or data.ndim == 4)

    # Initialize the output (copy)
    output = data.astype(np.float32)

    # Remove the local mean
    # The 4th dimension is the number of features
    if data.ndim == 4:
        for n in range(data.shape[3]):
            output[:, :, :, n] = output[:, :, :, n] - gaussian_filter(output[:, :, :, n], filter_width)
        # Compute the standard deviation locally across features and average
        stddev = np.sqrt(gaussian_filter(np.sum(output**2, axis=3), filter_width))
        # Estimate signal floor
        _, _, uf = noise_stats(stddev)
        # Robust division
        for n in range(data.shape[3]):
            output[:, :, :, n] = output[:, :, :, n]  * stddev / (stddev ** 2 + uf ** 2)
    else:
        output = output - gaussian_filter(output, filter_width)
        # Compute the standard deviation locally
        stddev = np.sqrt(gaussian_filter(output**2, filter_width))
        # Estimate signal floor
        _, _, uf = noise_stats(stddev)
        # Robust division
        output = output * stddev / (stddev**2 + uf**2)

    return output
