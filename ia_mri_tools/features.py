import numpy as np
from .signal_stats import noise_stats
from .filters import radial, gradient, hessian

NSCALES = 4
NORM_SCALE = 4


def _pinv(x, lam=3):
    """
    Pseudoinverse with regularization
    """
    _, q, _ = noise_stats(np.abs(x))
    ix = x / (np.abs(x)**2 + lam**2*q**2)
    return ix

def _get_dim(data):
    """
    Data dimensionality with error checking
    """

    if data.ndim == 2:
        ndim = 2
    elif data.ndim == 3:
        ndim = 3
    else:
        raise RuntimeError('Unsupported number of dimensions {}. We only supports 2 or 3D arrays.'.format(data.ndim))
    
    return ndim


def high_pass(data, norm=True, norm_scale=NORM_SCALE):
    """
    High pass filter
    """
    hp = data - radial(data, 'gaussian', scale=0)

    if norm:
        scale = np.sqrt(_pinv(radial(np.abs(hp)**2, 'gaussian', scale=norm_scale)))
        hp = hp * scale

    return hp


def gauss(data, scale=0, norm=True, norm_scale=NORM_SCALE):
    """
    Zeroth order gaussian derivative (rotationally invariant)
    """
    g = radial(data, 'gaussian', scale=scale)

    if norm:
        scale = np.sqrt(_pinv(radial(np.abs(g)**2, 'gaussian', scale=norm_scale)))
        g *= scale

    return g


def grad(data, scale=0, norm=True, norm_scale=NORM_SCALE):
    """
    Rotational invariant of the first order gaussian derivative
    """
    ndim = _get_dim(data)

    grad = gradient(data, scale=scale)
    # [dx, dy], or
    # [dx, dy, dz]

    # Local power normalization
    if norm:
        if ndim == 2:
            grad_pow = np.abs(grad[0])**2 + np.abs(grad[1])**2
        else:
            grad_pow = np.abs(grad[0])**2 + np.abs(grad[1])**2 + np.abs(grad[2])**2
        n = np.sqrt(_pinv(radial(grad_pow, 'gaussian', scale=norm_scale)))
        grad = [x*n for x in grad]

    # The rotaional invariant is the vector magnitude
    if ndim == 2:
        g = np.sqrt(np.abs(grad[0])**2 + np.abs(grad[1])**2)
    else:
        g = np.sqrt(np.abs(grad[0])**2 + np.abs(grad[1])**2 + np.abs(grad[2])**2)

    return g


def hess(data, scale=0, norm=True, norm_scale=NORM_SCALE):
    """
    Rotational invariant of the second order gaussian derivative
    """
    ndim = _get_dim(data)

    h = hessian(data, scale=scale)

    # Local power normalization
    if norm:
        if ndim == 2:
            # [dxx, dxy, dyy]
            hess_pow = np.abs(h[0])**2 + 2*np.abs(h[1])**2 + np.abs(h[2])**2
        else:
            # [dxx, dxy, dxz, dyy, dyz, dzz]
            hess_pow = np.abs(h[0])**2 + 2*np.abs(h[1])**2 + 2*np.abs(h[2])**2 + np.abs(h[3])**2 + 2*np.abs(h[4])**2 + np.abs(h[5])**2
        n = np.sqrt(_pinv(radial(hess_pow, 'gaussian', scale=norm_scale)))
        h = [x*n for x in h]

    # Rotational invariants and Frobenius norm
    if ndim == 2:
        # [dxx, dxy, dyy]
        # 1st trace l1 + l2
        trace = h[0] + h[2]
        # 2nd determinant l1*l2
        # det = Dxx*Dyy - Dxy*Dyx
        det = h[0]*h[2] - h[1]*h[1]
        # frobenius norm sqrt(l1**2 + l2**2)
        frobenius = np.sqrt(np.abs(h[0])**2 + 2*np.abs(h[1])**2 + np.abs(h[2])**2)

        return (trace, det, frobenius)

    else:
        # [dxx, dxy, dxz, dyy, dyz, dzz]
        # 1st trace l1 + l2 + l3
        trace = h[0] + h[3] + h[5]
        # 2nd l1*l2 + l1*l3 + l2*l3
        # sec = Dxx*Dyy - Dxy*Dyx + Dxx*Dzz - Dxz*Dzx + Dyy*Dzz - Dyz*Dzy
        sec = h[0]*h[3] - h[1]*h[1] + h[0]*h[5] - h[2]*h[2] + h[3]*h[5] - h[4]*h[4]
        # 3rd determinant l1*l2*l3
        # det = Dxx*(Dyy*Dzz - Dyz*Dzy) - Dxy*(Dyx*Dzz - Dyz*Dzx) + Dxz*(Dyx*Dzy - Dyy*Dzx)
        det = h[0]*(h[3]*h[5]-h[4]*h[4]) - h[1]*(h[1]*h[5]-h[4]*h[2]) + h[2]*(h[1]*h[4]-h[3]*h[2])
        # frobenius norm sqrt(l1**2 + l2**2 + l3**2)
        frobenius = np.sqrt(np.abs(h[0])**2 + 2*np.abs(h[1])**2 + 2*np.abs(h[2])**2 + np.abs(h[3])**2 + 2*np.abs(h[4])**2) + np.abs(h[5])**2

        return (trace, sec, det, frobenius)


def textures(data, nscales=NSCALES, norm_scale=NORM_SCALE):
    """
    Compute multi-scale rotationally invariant image features
    from the zeroth, first, and second order gaussian derivatives
    with divisive normalization
    
    :param data:  2D or 3D numpy array
    :param nscales: int number of scales
    :return: list of 3*nscales+1 2D or 3D numpy arrays and list of names
    """

    # Initialize the textures list and the names list
    t = []
    names = []

    # The first feature is the high pass filter
    feat = high_pass(data, norm=True, norm_scale=norm_scale)
    t.append(feat)
    names.append(f'High Pass')
    
    # The next set of features are the rotational invariants of the zeroth order gaussian derivatives
    for lev in range(nscales):
        feat = gauss(data, scale=lev, norm=True, norm_scale=norm_scale)
        t.append(feat)
        names.append(f'Gaussian S{lev}')

    # The next set of features are the rotational invariants of the first order gaussian derivatives
    for lev in range(nscales):
        feat = grad(data, scale=lev, norm=True, norm_scale=norm_scale)
        t.append(feat)
        names.append(f'Gradient S{lev}R1')

    # The next set of features are the rotational invariants of the second order gaussian derivatives
    for lev in range(nscales):
        hfeat = hess(data, scale=lev, norm=True, norm_scale=norm_scale)
        for n in range(len(hfeat)):
            t.append(hfeat[n])
            names.append(f'Hessian S{lev}R{n+1}')

    return t, names
