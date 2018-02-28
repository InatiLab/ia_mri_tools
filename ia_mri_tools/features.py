import numpy as np
from scipy.ndimage import gaussian_filter, gaussian_gradient_magnitude, gaussian_laplace


def mean(data, scale):
    """Compute image mean at a particular scale

    gaussian smoothing

    :param data:  2D or 3D numpy array
    :param scale: int
    :return: 2D or 3D numpy float32 array
    """
    d = data.astype(np.float32)
    m = gaussian_filter(d, sigma=scale)

    return m


def stddev(data, scale):
    """Compute image standard deviation at a particular scale

    gaussian smoothing

    :param data:  2D or 3D numpy array
    :param scale: int
    :return: 2D or 3D numpy float32 array
    """
    d = data.astype(np.float32)
    s = np.sqrt(gaussian_filter((d-mean(d, scale))**2, sigma=scale))

    return s


def grad(data, scale):
    """Compute image gradient magnitude at a particular scale

    gaussian smoothing

    :param data:  2D or 3D numpy array
    :param scale: int
    :return: 2D or 3D numpy float32 array
    """
    d = data.astype(np.float32)
    g = gaussian_gradient_magnitude(d, sigma=scale)

    return g


def laplace(data, scale):
    """Compute image laplacian at a particular scale

    gaussian smoothing

    :param data:  2D or 3D numpy array
    :param scale: int
    :return: 2D or 3D numpy float32 array
    """
    d = data.astype(np.float32)
    g = gaussian_laplace(d, sigma=scale)

    return g


def textures(data, scales=5, basename=''):
    """Compute image textures at a particular scale or set of scales
    gaussian smoothing, gradient magnitude, and standard deviation

    :param data:  2D or 3D numpy array
    :param scales: int or list of ints
    :param basename: str basename for feature labels
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
        t[..., 4*s+1] = mean(d, scales[s])
        names.append('{}_mean_{}'.format(basename, scales[s]))
        # gradient magnitude
        t[..., 4*s+2] = grad(d, scales[s])
        names.append('{}_grad_{}'.format(basename, scales[s]))
        # laplacian
        t[..., 4*s+3] = laplace(d, scales[s])
        names.append('{}_lap_{}'.format(basename, scales[s]))
        # standard deviation
        t[..., 4*s+4] = stddev(data, scales[s])
        names.append('{}_stddev_{}'.format(basename, scales[s]))

    return t, names
