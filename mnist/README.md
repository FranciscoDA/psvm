
# Place MNIST dataset here!

expected filename       | expected contents
------------------------|--------------------
train-images-idx3-ubyte | mnist train images (unsigned byte format)
train-labels-idx1-ubyte | mnist train labels (unsigned byte format)
t10k-images-idx3-ubyte  | mnist test images (unsigned byte format)
t10k-labels-idx1-ubyte  | mnist test labels (unsigned byte format)

The MNIST dataset by LeCun et al. can be found at http://yann.lecun.com/exdb/mnist/

# Otherwise, run get.sh to fetch & unzip it
```./get.sh```


The training dataset contains a collection of 60.000 28x28 grayscale pictures of handwritten digits. The test dataset contains 10.000 samples instead.
