import math as m
from logging import getLogger
logger = getLogger(__name__)

import numpy as np
from scipy.interpolate import griddata

def interpolate_image(v):
    x, y = np.meshgrid(range(v.shape[1]), range(v.shape[0]))
    x = x.flatten()
    y = y.flatten()
    z = v.flatten()
    points = np.stack([x[np.logical_not(np.isnan(z))], y[np.logical_not(np.isnan(z))]], axis=1)
    values = z[np.logical_not(np.isnan(z))]
    return np.reshape(griddata(points, values, (x, y), method='cubic'), v.shape).astype(np.float32)

def normalize(proj, *, bg=None, dark=None):
    if bg is not None:
        bg = np.log(bg - dark) if dark is not None else np.log(bg)
        proj = bg - (np.log(proj - dark) if dark is not None else np.log(proj))
    else:
        bg = np.full(proj.shape, m.log(2.0 ** 16 - 1))
        proj = bg - (np.log(proj - dark) if dark is not None else np.log(proj))

    if np.any(np.isnan(bg)): 
        logger.warn('X-ray has too low intensity in some region.')
        proj[np.isnan(bg)] = 0.0
    if np.any(np.isnan(proj)): 
        logger.warn('X-ray was not transmitted in some region.')
        proj = interpolate_nan(proj)

    return proj

