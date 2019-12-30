import os
import numpy as np
import numpy.ctypeslib as npct
import ctypes as c

array2d = npct.ndpointer(dtype=np.float32, ndim=2, flags='CONTIGUOUS')
array1d = npct.ndpointer(dtype=np.float32, ndim=1, flags='CONTIGUOUS')

libct = npct.load_library("libct.so", os.path.dirname(__file__))

libct.fbp.restype = None
libct.fbp.argtypes = [array2d, array2d, c.c_int, c.c_int, array1d, c.c_float]

def run(sino, angles, *, center=None):
    assert (sino.ndim == 2), "Invalid function argument: input 2-dimensional array."
    n_angles, width = sino.shape

    if sino.dtype != np.float32:
        sino = sino.astype(np.float32)
    if center is None:
        center = width / 2

    assert angles.shape == (n_angles,), ("Invalid function argument: {} != {}".format(n_angles, angles.shape))
    assert (0 < center and center < width), ("Invalid function argument: center is restricted in a range (0, {})".format(width))
    ct = np.zeros((width, width), dtype=np.float32)
    libct.fbp(sino, ct, width, n_angles, angles, center)
    return ct

