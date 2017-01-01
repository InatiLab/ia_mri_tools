# -*- coding: utf-8 -*-
import numpy as np
import scipy as sp
import logging

logger = logging.getLogger(__name__)

# add a console handler
ch = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)


def noise_level(data, tol=1e-2):

    d = data[data>0].flatten()

    # find the quartiles of the non-zero data
    Q1, Q2, Q3 = np.percentile(d,[25,50,75])
    logger.debug('Quartiles of the original data: {}, {}, {}'.format(Q1 , Q2, Q3))

    # find the quartiles of the non-zero data that is less than a cutoff
    # start with the first quartile and then iterate using the upper fence
    UF = Q1

    # repeat
    for iter in range(20):
        Q1, Q2, Q3 = np.percentile(d[d<UF], [25,50,75])
        logger.debug('Iteration {}. Quartiles of the trimmed data: {}, {}, {}'.format(iter, Q1 , Q2, Q3))
        Q13 = Q3 - Q1
        UFK = Q2 + 1.5*Q13
        # check for convergence
        if abs(UFK-UF)/UF < tol:
            UF = UFK
            break
        else:
            UF = UFK
    else:
        logger.warning('Warning, number of iterations exceeded')

    # recompute the quartiles
    Q1, Q2, Q3 = np.percentile(d[d<UF], [25,50,75])
    Q13 = Q3-Q1
    # Q1, Q2, Q3 describes the noise
    # anything above this is a noise outlier above (possibly signal)
    UF = Q2 + 1.5*Q13
    # anything below LF is a signal outlier below (not useful)
    LF = Q2 - 1.5*Q13
    # but remember that the noise distribution is NOT symmetric, so UF is an underestimate

    return LF, Q2, UF


def signal_likelihood(data, noise=None):
    """Return a likelihood that data is signal

    in SNR units, sigmoid with width 1, shifted to the right by 1
    ie P(<1)=0, P(2)=0.46, P(3)=0.76, P(4)=0.01, P(5)=0.96
    """

    if not noise:
        _, _, noise = noise_level(data)

    # The probability that each point has a signal
    P = (data>noise) * (-1 + 2 / (1+np.exp(-(data-noise)/noise)))

    return P
