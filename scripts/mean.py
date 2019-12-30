import os
import argparse
import numpy as np
import tifffile as tiff
#import tqdm

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output", help="file name for output", default="output.tif")
    parser.add_argument("input_dir")
    args = parser.parse_args()

    #print(args.input_dir)
    #print(args.output)

    dirname = args.input_dir
    filename = args.output

    assert (os.path.exists(dirname) and os.path.isdir(dirname)), ("Input directory does not exist: {}".format(dirname))
    assert (not os.path.exists(filename) or not os.path.isdir(filename)), ("Directory exists: {}".format(filename))

    files = list(map(lambda f:
            os.path.join(dirname, f), os.listdir(dirname)))
    files.sort()

    images = tiff.imread(files)
    num, _, _ = images.shape
    mean = np.sum(images, axis=0) / num

    tiff.imwrite(filename, mean.astype(np.float32))

#if __name__ == "__main__" : main()
main()

