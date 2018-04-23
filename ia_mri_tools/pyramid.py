import math
from math import floor, ceil
import numpy as np
from numpy.fft import fftn, ifftn, fftshift, ifftshift

from .filters import radial, gradient, hessian

# Default number of pyramid levels
NLEVELS = 4


##################
#    Pyramids    #
# ################

# Gaussian Pyramid
def gaussian_pyramid(data, nlevels=NLEVELS):
    return [radial(data, 'gaussian', level) for level in range(nlevels)]


# Difference of Gaussians Pyramid
def dog_pyramid(data, nlevels=NLEVELS):
    return [radial(data, 'dog', level) for level in range(nlevels)]


# Laplacian Pyramid
def laplacian_pyramid(data, nlevels=NLEVELS):
    return [radial(data, 'laplacian', level) for level in range(nlevels)]


# Gradient Pyramid
def gradient_pyramid(data, nlevels=NLEVELS):
    output = []
    for level in range(nlevels):
        output.append(gradient(data, level))
    return output


# Hessian Pyramid
def hessian_pyramid(data, nlevels=NLEVELS):
    output = []
    for level in range(nlevels):
        output.append(hessian(data, level))
    return output


###################
#    Utilities    #
# #################

# Up/Down sampling utilities
def nearest_multiple(n, m):
    """The multiple of m closest to n from above
    """
    return ceil(n/m) * m


def zeropad(data, m=2):
    """Pad with zeros to make an array with an even multiple of m voxels in all dimensions
    """

    if data.ndim < 2 or data.ndim > 3:
        raise RuntimeError('Unsupported number of dimensions {}.  We only supports 2 or 3D arrays.'.format(data.ndim))
        
    if data.ndim == 2:
        nx, ny = data.shape
        nxo = nearest_multiple(nx, 2*m)
        nyo = nearest_multiple(ny, 2*m)
        output = np.zeros([nxo, nyo], data.dtype)
        output[:nx, :ny] = data
        
    else:
        nx, ny, nz = data.shape
        nxo = nearest_multiple(nx, 2*m)
        nyo = nearest_multiple(ny, 2*m)
        nzo = nearest_multiple(nz, 2*m)
        output = np.zeros([nxo, nyo, nzo], data.dtype)
        output[:nx, :ny, :nz] = data

    return output


def downsample(data):
    """Downsample by 2 in every dimension in the fourier domain
    """
    
    if data.ndim < 2 or data.ndim > 3:
        raise RuntimeError('Unsupported number of dimensions {}.  We only supports 2 or 3D arrays.'.format(data.ndim))
        
    temp = fftshift(fftn(data))

    if data.ndim == 2:
        nx, ny = data.shape
        nxs, nxe = floor(nx/4), floor(3*nx/4)
        nys, nye = floor(ny/4), floor(3*ny/4)
        output = temp[nxs:nxe, nys:nye]

    else:
        nx, ny, nz = data.shape
        nxs, nxe = floor(nx/4), floor(3*nx/4)
        nys, nye = floor(ny/4), floor(3*ny/4)
        nzs, nze = floor(nz/4), floor(3*nz/4)
        
        output = temp[nxs:nxe, nys:nye, nzs:nze]

    output = ifftn(ifftshift(output))

    if np.isrealobj(data):
        return np.real(output)
    else:
        return output


def upsample(data):
    """Upsample by 2 in every dimension in the fourier domain
    """
    
    if data.ndim < 2 or data.ndim > 3:
        raise RuntimeError('Unsupported number of dimensions {}.  We only supports 2 or 3D arrays.'.format(data.ndim))
    
    temp = fftshift(fftn(data))

    if data.ndim == 2:
        nx, ny = data.shape
        nxo, nyo = 2*nx, 2*ny
        nxs, nxe = floor(nxo/4), floor(3*nxo/4)
        nys, nye = floor(nyo/4), floor(3*nyo/4)

        output = np.zeros([nxo, nyo], temp.dtype)
        output[nxs:nxe, nys:nye] = temp

    else:
        nx, ny, nz = data.shape
        nxo, nyo, nzo = 2*nx, 2*ny, 2*nz
        nxs, nxe = floor(nxo/4), floor(3*nxo/4)
        nys, nye = floor(nyo/4), floor(3*nyo/4)
        nzs, nze = floor(nzo/4), floor(3*nzo/4)
        output = np.zeros([nxo, nyo, nzo], temp.dtype)
        output[nxs:nxe, nys:nye, nzs:nze] = temp

    output = ifftn(ifftshift(output))

    if np.isrealobj(data):
        return np.real(output)
    else:
        return output
