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
            return UFK
        else:
            UF = UFK

    logger.warning('Warning, number of iterations exceeded')

    return UF
