import numpy as np

from .pyramid import laplacian_pyramid, gradient_pyramid

NLEVELS = 4

def textures(data, nlevels=NLEVELS):
    """Compute rotationally invariant image textures at a particular scale or set of scales
    original data, laplacian pyramid, mag of laplacian pyramid, mag of gradient pyramid

    :param data:  2D or 3D numpy array
    :param nlevels: int number of levels in each of the pyramids
    :return: list of 2D or 3D numpy arrays and list of feature names
    """

    # Initialize the textures list and the names list
    t = []
    names = []

    # The first texture is the original data
    t.append(data)
    names.append('Original Data')

    # Then the laplacian pyramid
    lap_pyr = laplacian_pyramid(data, nlevels)
    for level in range(nlevels):
        t.append(lap_pyr[level])
        names.append('Laplacian level {}'.format(level))
 
    # And its magnitude
    for level in range(nlevels):
        t.append(np.abs(lap_pyr[level]))
        names.append('Magnitude of the Laplacian level {}'.format(level))

    # Then the magnitude of the gradient pyramid
    grad_pyr = gradient_pyramid(data, nlevels)
    norientations = data.ndim
    for level in range(nlevels):
        temp = np.zeros(data.shape, data.dtype)
        for orient in range(norientations):
            temp += np.abs(grad_pyr[level][orient]**2)
        t.append(np.sqrt(temp))
        names.append('Magnitude of the Gradient level {}'.format(level))
    
    return t, names
