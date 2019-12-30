import os
import argparse
import numpy as np
import tifffile as tiff
from tqdm import tqdm
import fbp

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output", help="file name for output", default="ct")
    parser.add_argument("-s", "--slice", help="specify slice number if you want a single ct", type=int)
    parser.add_argument("-c", "--center", help="specify slice number if you want a single ct", type=float)
    parser.add_argument("--i0", help="file name for i0 iamge", default="i0.tif")
    parser.add_argument("--dark", help="file name for dark iamge", default="dark.tif")
    parser.add_argument("input_dir")
    args = parser.parse_args()

    #print(args.input_dir)
    #print(args.output)

    in_dir = args.input_dir
    out_dir = args.output
    i0_file = args.i0
    dark_file = args.dark

    assert (os.path.exists(in_dir) and os.path.isdir(in_dir)), ("Input directory does not exist: {}".format(in_dir))
    assert (not os.path.exists(out_dir) or os.path.isdir(out_dir)), ("File exists: {}".format(out_dir))
    assert (os.path.exists(i0_file) ), ("File does not exist: {}".format(i0_file))
    assert (os.path.exists(dark_file) ), ("File does not exist: {}".format(dark_file))

    files = list(map(lambda f:
            os.path.join(in_dir, f), os.listdir(in_dir)))
    files.sort()

    dark = tiff.imread(dark_file).astype(np.float32)
    i0 = np.log(tiff.imread(i0_file).astype(np.float32) - dark)

    images = tiff.imread(files).astype(np.float32)
    n_angles, height, width = images.shape

    print("normalize")
    for i in tqdm(range(0, n_angles)):
        images[i,:,:] = i0 - np.log(images[i,:,:] - dark)

    tiff.imwrite("sino.tif", images[:,int(height/2),:])

    angles = (np.pi / float(n_angles)) * np.arange(0, n_angles, dtype=np.float32)

    if args.slice is not None:
        ct = fbp.run(images[:,args.slice,:].copy(), angles, center=args.center)
        tiff.imwrite("{}_{}.tif".format(out_dir, args.slice), ct)
    else:
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)
        print("reconstruct")
        for i in tqdm(range(0, height)):
            ct = fbp.run(images[:,i,:].copy(), angles, center=args.center)
            tiff.imwrite(os.path.join(out_dir, "a{:04d}.tif".format(i)), ct)

main()

