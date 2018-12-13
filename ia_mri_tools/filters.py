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
        raise RuntimeError('Unsupported number of dimensions {}. We only supports 2 or 3D arrays.'.format(len(shape)))


def _pinv(x, p=2):
    """
    Pseudoinverse with regularization

    Use the lowest p percent of the signal as an estimate of the noise floor
    """
    d = np.abs(x)
    # find the lowest p% of the non-zero signal
    # to use for regularization
    s = np.percentile(d[d>0], [p])
    ix = x / (d**2 + s**2)
    return ix


def radial(data, func, scale=1, truncate=False):
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


def high_pass(data, scale=0):
    """
    High pass filter
    """
    hp = data - radial(data, 'gaussian', scale=scale)

    return hp


def gradient(data, scale=1, truncate=False):
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


def hessian(data, scale=1, truncate=False):
    """                                                                                                                                                              
    Hessian, 2nd order partial derivatives filter in the fourier domain with truncation                                                                                                            
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
    # (i*x)*(i*y)*g, etc
    temp = -1.0 * g * fftshift(fftn(data))
    dxx = ifftn(ifftshift(x*x*temp))
    dxy = ifftn(ifftshift(x*y*temp))
    dyy = ifftn(ifftshift(y*y*temp))
    if data.ndim == 3:
        dxz = ifftn(ifftshift(x*z*temp))
        dyz = ifftn(ifftshift(y*z*temp))
        dzz = ifftn(ifftshift(z*z*temp))

    # Ensure that real functions stay real                                                                                                                           
    if np.isrealobj(data):
        dxx = np.real(dxx)
        dxy = np.real(dxy)
        dyy = np.real(dyy)
        if data.ndim == 3:
            dxz = np.real(dxz)
            dyz = np.real(dyz)
            dzz = np.real(dzz)

    if data.ndim == 2:
        return [dxx, dxy, dyy]
    elif data.ndim == 3:
        return [dxx, dxy, dxz, dyy, dyz, dzz]
    else:
        raise RuntimeError('Unsupported number of dimensions {}.'.format(data.ndim))


def hessian_power(h):
    """
    Power a the hessian filter band
    Frobenius norm squared
    """
    if len(h) == 2:
        p = np.abs(h[0])**2 + 2*np.abs(h[1])**2 + np.abs(h[2])**2
    elif len(h) == 6:
        p = np.abs(h[0])**2 + 2*np.abs(h[1])**2 + 2*np.abs(h[2])**2 + np.abs(h[3])**2 + 2*np.abs(h[4])**2 + np.abs(h[5])**2
    else:
        raise RuntimeError('Unsupported number of dimensions {}.'.format(len(h)))
    return p


def gradient_rot(g):
    """
    Rotational invariant of the gradient
    """
    if len(g) == 2:
        # [dx, dy]    
        g = np.sqrt(np.abs(g[0])**2 + np.abs(g[1])**2)

    elif len(g) == 3:
        # [dx, dy, dz]    
        g = np.sqrt(np.abs(g[0])**2 + np.abs(g[1])**2 + np.abs(g[2]))

    else:
        raise RuntimeError('Unsupported number of dimensions {}.'.format(len(g)))

    return g

def hessian_rot(h):
    """
    Rotational invariants of the hessian
    """

    if len(h) == 3:
        # [dxx, dxy, dyy]
        # 1st trace l1 + l2
        trace = h[0] + h[2]
        # 2nd determinant l1*l2
        # det = Dxx*Dyy - Dxy*Dyx
        det = h[0]*h[2] - h[1]*h[1]
        # frobenius norm sqrt(l1**2 + l2**2)
        frobenius = np.sqrt(np.abs(h[0])**2 + 2*np.abs(h[1])**2 + np.abs(h[2])**2)
        # normalize the determinant by the frobenius norm for scaling
        det = det * _pinv(frobenius)

        return (trace, det, frobenius)

    elif len(h) == 6:
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

        # normalize the second and third rotational invariants by the frobenius norm for scaling
        sec = sec * _pinv(frobenius)
        det = det * _pinv(frobenius**(3/2))

        return (trace, sec, det, frobenius)

    else:
        raise RuntimeError('Unsupported number of dimensions {}.'.format(len(h)))
