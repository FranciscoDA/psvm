# psvm
Sequential and parallel (CUDA) implementations of SVM classifiers.

## Requirements
 * CMake 3.8 (required for native CUDA support)
 * Boost 1.60
 * CUDA SDK
 * C++14 compiler

## Compilation
 * Clone this repository: `git clone https://github.com/franciscoda/psvm`
 * Create build directory: `mkdir build && cd build`
 * Run CMake and (optionally) specify a GPU architecture: `cmake .. -DCMAKE_CUDA_FLAGS="-arch=sm30"`
 * Compile: `make`
 * The output paths for the CLI executables executables are `build/bin/svm` and `build/bin/cusvm`
 * The output paths for the static libraries are `build/lib/libsvm.a` and `build/lib/libcusvm.a`

## Features

Run `bin/svm --help` or `bin/cusvm --help` to see a list of parameters.

### Supported input formats
 * CSV
 * IDX (see the MNIST handwritten digit dataset format)

Note that attributes and labels must be in separate files.

### Supported normalization methods:
 * `--nz`: (x-mean)/stdev. Each attribute is scaled according to the normal distribution
 * `--n1`: (x-min)/(max-min). Each attribute is scaled to \[0;1\] range
 * `--n2`: -1 + (x-min)\*(1-(-1))/(max-min). Each attribute is scaled to \[-1;1\] range

### Supported kernels (--kernel flag)
 * `--linear`: xi \* xj
 * `--polynomial`: (xi\*xj+c)^d
 * `--gaussian`: exp(-gamma * ||xi-xj||^2)

### Supported multiclass classification methods
 * `--1A1` - One against one (trains k\*(k-1)/2 reduced models)
 * `--1AA` - One against all (trains k models)

## References / Recommended reads:
 * [Buttou, L., Lin, C. (2006). *Support Vector Machine Solvers*.](http://leon.bottou.org/publications/pdf/lin-2006.pdf)
 * [Burges, C. (1998). *A Tutorial on support Vector Machines for Pattern Recognition*.](http://www.di.ens.fr/~mallat/papiers/svmtutorial.pdf)
 * [Chang, C., Lin, C. (2001). *LIBSVM: A Library for Support Vector Machines*.](https://www.csie.ntu.edu.tw/~cjlin/papers/libsvm.pdf)
 * [Berwick, R. *An Idiot's Guide to Support Vector Machines (SVMs).*](http://web.mit.edu/6.034/wwwbob/svm-notes-long-08.pdf)
