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


def coil_correction(data, width=10, scale=100):
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

    # Find the signal statistics
    lf, q2, uf = noise_stats(data)

    # Weights
    w = data**2 / (data**2 + uf**2)

    # Smooth estimates of data and data**2
    u1 = gaussian_filter(w*data, width)
    u2 = gaussian_filter(w*data**2, width)

    # Coil map (soft inverse)
    c = u1 * u2 / (u2**2 + uf**4)

    # and scale
    c *= scale

    return c


def textures(data, scales=5, basename='', whiten=True, mask=None):
    """Compute image textures at a particular scale or set of scales
    gaussian smoothing, gradient magnitude, laplacian and standard deviation

    :param data:  2D or 3D numpy array
    :param scales: int or list of ints
    :param basename: str basename for feature labels
    :param whiten: bool mean center and variance normalize each feature
    :param mask: 2D or 3D numpy array or None Mask for whitening
    :return: 3D or 4D numpy float32 array, list of feature labels
    """

    if isinstance(scales, int):
        scales = [scales]

    ns = len(scales)
    out_shape = list(data.shape)
    out_shape.append(4*ns+1)
    t = np.zeros(out_shape, dtype=np.float32)
    d = data.astype(np.float32)

    # the first texture is the original data
    t[..., 0] = d
    names = [basename]

    # loop over scales
    for s in range(ns):
        # mean
        t[..., 4*s+1] = gaussian_filter(d, sigma=scales[s])
        names.append('{}_mean_{}'.format(basename, s))
        # gradient magnitude
        t[..., 4*s+2] = gaussian_gradient_magnitude(d, sigma=scales[s])
        names.append('{}_gradient_{}'.format(basename, s))
        # laplacian
        t[..., 4*s+3] = gaussian_laplace(d, sigma=scales[s])
        names.append('{}_laplacian_{}'.format(basename, s))
        # standard deviation
        t[..., 4*s+4] = np.sqrt(gaussian_filter((d - t[..., 4*s+1])**2, sigma=scales[s]))
        names.append('{}_deviation_{}'.format(basename, s))

    if whiten:
        for q in range(t.shape[-1]):
            vol = t[..., q]
            if mask is not None:
                vol = vol[mask]
            m = np.mean(vol.flatten())
            s = 1.0/np.std(vol.flatten())
            t[..., q] = s * (t[..., q] - m)

    return t, names


def select(data, mask=None):

    if isinstance(data, list):
        h = []
        for dsub in data:
            h.append(select(dsub, mask))
        return np.hstack(h)
    else:
        if mask is not None:
            if len(data.shape) == 3:
                return data.reshape(-1, 1)[mask.flatten(), :]
            else:
                return data.reshape(-1, data.shape[-1])[mask.flatten(), :]
        else:
            if len(data.shape) == 3:
                return data.reshape(-1, 1)
            else:
                return data.reshape(-1, data.shape[-1])
