# psvm
Sequential and parallel (CUDA) implementations of SVM classifiers.

## Features

Run `./svm --help` or `./cusvm --help` to see a list of parameters.

### Supported input formats
 * CSV
 * IDX (see the MNIST handwritten digit dataset format)

### Supported kernels
 * Linear: xi \* xj
 * Polynomial: (xi\*xj+c)^d
 * Gaussian: exp(-||xi-xj||^2/gamma)

### Supported multiclass classification methods
 * 1A1 - One against one (trains k\*(k-1)/2 reduced models)
 * 1AA - One against all (trains k models)

## Compilation
 * Clone this repository
 * Download submodules
   * `$ git submodule init`
   * `$ git submodule update`
 * `$ make`

## References:
 * Buttou, L., Lin, C. (2006). Support Vector Machine Solvers
 * Burges, C. (1998). A Tutorial on support Vector Machines for Pattern Recognition
