
# Place MNIST dataset here!

expected filename       | expected contents
------------------------|--------------------
train-images-idx3-ubyte | mnist train images (unsigned byte format)
train-labels-idx1-ubyte | mnist train labels (unsigned byte format)
t10k-images-idx3-ubyte  | mnist test images (unsigned byte format)
t10k-labels-idx1-ubyte  | mnist test labels (unsigned byte format)

The MNIST dataset by LeCun et al. can be found at http://yann.lecun.com/exdb/mnist/

The training dataset contains a collection of 60.000 28x28 grayscale pictures of handwritten digits. The test dataset contains 10.000 samples instead. All pictures are in binary IDX format.

A few scripts have been included for convenience:
 * `fetch.sh` downloads and unzips the dataset
 * `idx2png`: extracts a series of samples from an input dataset in idx format and outputs a set of PNG images to a directory. Sample usage: `idx2png.py train-images-idx3-ubyte dump/ $(seq 1 100)`. Requires Python3, Pillow and numpy. RGB and grayscale data is supported, but only in unsigned byte format.
 * `idx2csv`: extracts a series of samples from an input dataset in idx format and outputs a csv file with the flattened samples. Sample usage: `idx2csv.py train-images-idx3-ubyte dump.csv $(seq 1 100)`. Requires Python3 and numpy.

