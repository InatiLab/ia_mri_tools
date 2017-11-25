import numpy as np
from scipy.ndimage import gaussian_filter, gaussian_gradient_magnitude, gaussian_laplace


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
