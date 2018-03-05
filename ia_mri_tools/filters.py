import math
import numpy as np
from numpy.fft import fftn, ifftn, fftshift, ifftshift


def scale_coordinates(shape, scale):
    
    ndim = len(shape)
    # Compute the scaled coordinate system
    if ndim == 2:
        nx, ny = shape
        fx, fy = np.meshgrid(np.linspace(-nx/2, nx/2, nx), np.linspace(-ny/2, ny/2, ny), indexing='ij')
        sx = nx / (2.0*math.pi) / 2**scale
        sy = ny / (2.0*math.pi) / 2**scale
        x = fx/sx
        y = fy/sy
        return x, y

    elif ndim == 3:
        nx, ny, nz = shape
        fx, fy, fz = np.meshgrid(np.linspace(-nx/2, nx/2, nx),
                                 np.linspace(-ny/2, ny/2, ny),
                                 np.linspace(-nz/2, nz/2, nz),
                                 indexing='ij')
        sx = nx / (2.0*math.pi) / 2**scale
        sy = ny / (2.0*math.pi) / 2**scale
        sz = nz / (2.0*math.pi) / 2**scale
        x = fx/sx
        y = fy/sy
        z = fz/sz

        return x, y, z

    else:
        raise RuntimeError('Unsupported number of dimensions {}. We only supports 2 or 3D arrays.'.format(data.ndim))


def radial(data, func, scale=1, truncate=True):
    """
    Rotationally symmetric filter in the fourier domain with truncation
    """
    
    known_filters = ['gaussian', 'dog', 'laplacian']
    if func.lower() not in known_filters:
        raise RuntimeError('Unsupported filter function error {}.  Must be one of {}.'.format(func, known_filters))

    # Get the scaled coordinate system, [-pi,pi]^d
    if data.ndim == 2:
        x, y = scale_coordinates(data.shape, scale)
        rsq = np.sqrt(x*x + y*y)
    elif data.ndim == 3:
        x, y, z = scale_coordinates(data.shape, scale)
        rsq = np.sqrt(x*x + y*y + z*z)
    else:
        raise RuntimeError('Unsupported number of dimensions {}. We only supports 2 or 3D arrays.'.format(data.ndim))
        
    # Compute filter as a function of radius
    if func.lower() == 'gaussian':
        # Gaussian, 0th Hermite, etc.
        g = np.exp(-0.5*rsq)
    elif func.lower() == 'dog':
        # Difference of Gaussians
        g = np.exp(-0.5*rsq) - np.exp(-0.5*(4.0*rsq))
    elif func.lower() == 'derivative':
        # Derivative of Gaussian, 1st Hermite
        g = -1.0/(math.pi**2) * (1.0 - math.pi**2 * rsq) * np.exp(-0.5*rsq)
    elif func.lower() == 'laplacian':
        # Laplacian of Gaussian, 2nd Hermite, Marr, Sombrero, Ricker, etc.
        g = -1.0/(math.pi**2) * (1.0 - math.pi**2 * rsq) * np.exp(-0.5*rsq)
    else:
        raise RuntimeError('Unkown filter function {}.'.format(func))

    # Truncate on a sphere of r=pi^2
    if truncate:
        g[rsq > math.pi**2] = 0.0

    # Apply the filter
    output = ifftn(ifftshift(g*fftshift(fftn(data))))

    # Ensure that real functions stay real
    if np.isrealobj(data):
        return np.real(output)
    else:
        return output


def gradient(data, scale=1, truncate=True):
    """
    Gradient filter in the fourier domain with truncation
    """
    
    # Get the scaled coordinate system, [-pi,pi]^d
    if data.ndim == 2:
        x, y = scale_coordinates(data.shape, scale)
        rsq = np.sqrt(x*x + y*y)
    elif data.ndim == 3:
        x, y, z = scale_coordinates(data.shape, scale)
        rsq = np.sqrt(x*x + y*y + z*z)
    else:
        raise RuntimeError('Unsupported number of dimensions {}. We only supports 2 or 3D arrays.'.format(data.ndim))

    # Build the filter in the fourier domain

    # 1) Gaussian
    g = np.exp(-0.5*rsq)

    # 2) Truncate on a sphere of r=pi
    if truncate:
        g[rsq > math.pi**2] = 0.0

    # 3) Gradient in each direction
    # i*x*g, i*y*g, i*z*g etc.
    temp = 1j * g * fftshift(fftn(data))
    gx = ifftn(ifftshift(x*temp))
    gy = ifftn(ifftshift(y*temp))
    if data.ndim == 3:
        gz = ifftn(ifftshift(z*temp))

    # Ensure that real functions stay real
    if np.isrealobj(data):
        gx = np.real(gx)
        gy = np.real(gy)
        if data.ndim == 3:
            gz = np.real(gz)

    if data.ndim == 2:
        return [gx, gy]
    elif data.ndim == 3:
        return [gx, gy, gz]
    else:
        raise RuntimeError('Unsupported number of dimensions {}.'.format(data.ndim))
