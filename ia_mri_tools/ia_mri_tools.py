# -*- coding: utf-8 -*-
import numpy as np
from scipy.ndimage import gaussian_filter

from .signal_stats import signal_likelihood
from . import FILTER_WIDTH, SIGNAL_SCALE


def apply_global_contrast_normalization(data, signal_mean, signal_std_dev, signal_width=6,
                                        signal_min=0, signal_max=2*SIGNAL_SCALE):
    """
    Apply global contrast normalization to an image

    :param data: input image
    :param signal_mean: mean of the signal
    :param signal_std_dev: standard deviation of the signal
    :param signal_width: width of the output in standard deviations
    :param signal_min: minimum value of the output
    :param signal_max: maximum value of the output
    :return: global contrast normalized image
    """

    z_scored_data = (data - signal_mean)/signal_std_dev
    gcn = z_scored_data + signal_width/2
    gcn = (signal_max - signal_min)/signal_width * gcn
    gcn[gcn < signal_min] = signal_min
    gcn[gcn > signal_max] = signal_max

    return gcn


def estimate_global_contrast_normalization(data):
    """
    Estimate global contrast normalization parameters for an image
    :param data: original data
    :return:
      center:  mean of the image weighted by the signal_likelihood
      width: standard deviation of the image weighted by the signal_likelihood
    """

    weight = signal_likelihood(data)
    center = np.sum(weight*data)/np.sum(weight)
    width = np.sqrt(np.sum(weight*(data-center)**2)/np.sum(weight))

    return center, width


def global_contrast_normalize(data, signal_min=0, signal_max=2*SIGNAL_SCALE, signal_width=3):
    """

    :param data:
    :param signal_min:
    :param signal_max:
    :param signal_width:

    :return:
    """

    mean, std_dev = estimate_global_contrast_normalization(data)
    gcn = apply_global_contrast_normalization(data, mean, std_dev, signal_width=signal_width,
                                              signal_min=signal_min, signal_max=signal_max)

    return gcn


def local_contrast_normalize(data, filter_width=FILTER_WIDTH):

    weights = signal_likelihood(data)
    weighted_mean = gaussian_filter(weights * data, filter_width) / gaussian_filter(weights, filter_width)
    diff = data - weighted_mean
    std_dev = np.sqrt(gaussian_filter(np.abs(diff)**2, filter_width) / gaussian_filter(weights, filter_width))
    gcn = diff / std_dev

    return gcn


