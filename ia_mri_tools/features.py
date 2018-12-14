import numpy as np
from .filters import _pinv, radial, high_pass, gradient, hessian, gradient_rot, hessian_rot


# Scales for first stage local gain control
# and output gain control
HIGH_PASS_SCALE = 4
NORMALIZATION_SCALE = 5

# Number of spatial scales for rotationally invariant features
NUM_SCALES = 3


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


def compute_local_normalization(data, scale=NORMALIZATION_SCALE):
    """
    Compute inverse of the local norm of a feature

    :param data: 2D or 3D amplitude image
    :param scale: spatial scale averagin filter
    :return: 2D or 3D normalization image 

    normalized_data = normalization * data
    """

    # Compute the local amplitude
    a = np.sqrt(radial(np.abs(data)**2, 'gaussian', scale=scale))

    # Gain is (1/a)
    n = _pinv(a)

    return n


def input_normalization(data, high_pass_scale=HIGH_PASS_SCALE, normalization_scale=NORMALIZATION_SCALE):
    """
    Compute local normalization for the input in a manner similar to the retina
    Normalize by the power of the high pass filtered image
    """

    hp = high_pass(data, scale=high_pass_scale)
    norm = compute_local_normalization(hp, scale=normalization_scale)
    output = norm*data
    return output


def riff(data, nscales=NUM_SCALES, high_pass_scale=HIGH_PASS_SCALE, normalization_scale=NORMALIZATION_SCALE):
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

    # Initialize the total power for final stage normalization
    total_power = 0.0

    # Stage 1, Local gain control
    d = input_normalization(data, 
                            high_pass_scale=high_pass_scale,
                            normalization_scale=normalization_scale,
                            )

    # Step 2, feature generation
    # The first feature is the high pass filter
    feat = high_pass(d, scale=0)
    t.append(feat)
    names.append(f'High Pass')
    total_power += np.abs(feat)**2

    for lev in range(nscales):
        # The next set of features are the rotational invariants of the zeroth order gaussian derivatives
        feat = radial(d, 'gaussian', scale=lev)
        t.append(feat)
        names.append(f'Gaussian S{lev}')
        total_power += np.abs(feat)**2

        # The next set of features are the rotational invariants of the first order gaussian derivatives
        g = gradient(d, scale=lev)
        feat = gradient_rot(g)
        t.append(feat)
        names.append(f'Gradient S{lev}R1')
        # The power is the square of the gradient amplitude
        total_power += feat**2

        # The next set of features are the rotational invariants of the second order gaussian derivatives
        h = hessian(d, scale=lev)
        feat = hessian_rot(h)
        # The power is the the square of the Frobenius norm, the last rotationally invariant feature
        total_power += feat[-1]**2
        for n in range(len(feat)):
            t.append(feat[n])
            names.append(f'Hessian S{lev}R{n+1}')

    # The final amplitude
    total_amplitude = np.sqrt(total_power)
    # Step 3, the final local normalization
    norm = compute_local_normalization(total_amplitude, scale=normalization_scale)
    # Loop over the features and scale them all
    for n in range(len(t)):
        t[n] = norm*t[n]

    return t, names
