# -*- coding: utf-8 -*-
import numpy as np
from scipy.ndimage import gaussian_filter

from . import FILTER_WIDTH, SIGNAL_SCALE
from .signal_stats import noise_stats, signal_likelihood


def coil_correction(data, width=FILTER_WIDTH, scale=SIGNAL_SCALE):
    """Weighted least squares estimate of the coil intensity correction

    :param data: 2D or 3D array
    :param width: gaussian filter width
    :param scale:
    :return: 2D or 3D array coil correction

    data_corr = coil_correction(data)*data

    The algorithm is based on the observation that
    c = <data>/<data**2> is a solution to
    <data * c> = <1> in a weighted least squares sense

    The scale factor multiplies the above, so <data * c> = s*<1>
    """

    _, sigma_noise, _ = noise_stats(data)
    weights = signal_likelihood(data)
    c = scale * gaussian_filter(weights * data, width) / gaussian_filter(weights * data**2 + sigma_noise**2, width)

    return c


def coil_correction_glasser(t1, t2, width=FILTER_WIDTH):
    """ Coil correction for a pair of t1 and t2 images

    This implements an algorithm similar to the one described by Glasser and Van Essen in
    https://doi.org/10.1523/JNEUROSCI.2180-11.2011
      c ~ 1/sqrt(t1*t2)

    :param t1: 2D or 3D array
    :param t2: 2D or 3D array
    :param width: gaussian filter width
    :return: 2D or 3D array coil correction

    data_corr = coil_correction(data)*data
    """

    h = np.sqrt(t1*t2)
    _, sigma, _ = noise_stats(h)
    w = signal_likelihood(h)
    c = gaussian_filter(w*h, width) / gaussian_filter(w*h**2 + 9*sigma**2, width)

    scale = np.sum(w*h) / np.sum(w*c*h)

    c *= scale

    return c
