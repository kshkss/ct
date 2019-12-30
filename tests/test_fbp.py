import os
import numpy as np
import tifffile as tiff
import fbp

sino = tiff.imread("sinogram.tif")
n_angles, width = sino.shape
angles = (np.pi / n_angles) * np.arange(0, n_angles, dtype=np.float32)

ct = fbp.run(sino, angles, center=256)

tiff.imwrite("fbp_result.tif", ct)
