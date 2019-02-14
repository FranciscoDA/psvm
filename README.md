# psvm
Sequential and parallel (CUDA) implementations of SVM classifiers.

This project is divided among three main subprojects:
 * **libpsvm**: library implementing support vector machines, classifiers, training and decision algorithms. libpsvm contains two static libraries: `libsvm` and `libcusvm`. While the former performs sequential training, the later may train in parallel using CUDA GPGPU.
 * **psvm-cli**: command line interface that allows for training, testing and prediction of classifier models with data from CSV or IDX (MNIST) datasets. `psvm-cli` generates two executables named svm and cusvm. The former links to `libsvm` and the later to `libcusvm`.
 * **psvm4r**: R package with bindings to libpsvm. Note that currently `psvm4r` only links with libsvm.
 

## Requirements
 * CMake 3.4+
 * Boost 1.55
 * C++11 compiler
 * CUDA 8.0 SDK (optional)
 * R with Rcpp package (optional)

## Compilation
 * Clone this repository: `git clone https://github.com/franciscoda/psvm`
 * Download git submodules: `git submodule init && git submodule update -f`
 * Create build directory: `mkdir build && cd build`
 * Run CMake from the build directory: `cmake ..`
   * Optionally, specify a CUDA architecture. If CMake<=3.8: `cmake .. -DCUDA_NVCC_FLAGS="-arch=sm_30"`, else if CMake>3.8: `cmake .. -DCMAKE_CUDA_FLAGS="-arch=sm_30"`
   * Optionally, generate position-independent code. This is required if you plan to build psvm4r, since psvm4r builds a shared library. `cmake .. -DCMAKE_POSITION_INDEPENDENT_CODE=ON`
 * Compile: `make`
 * The output paths for the CLI executables executables are `build/bin/svm` and `build/bin/cusvm`
 * The output paths for the static libraries are `build/lib/libsvm.a` and `build/lib/libcusvm.a`

## psvm-cli

psvm-cli provides most of the features implemented in libpsvm, plus some basic I/O and data preprocessing functions. Run `bin/svm --help` or `bin/cusvm --help` to see a list of parameters.

### Supported input formats
 * CSV
 * IDX (see the MNIST handwritten digit dataset format, also check the `mnist/` subdirectory)

Note that attributes and labels must be in separate files.

### Supported normalization methods:
 * `--nz`: (x-mean)/stdev. Each attribute is scaled according to the normal distribution
 * `--n1`: (x-min)/(max-min). Each attribute is scaled to \[0;1\] range
 * `--n2`: -1 + (x-min)\*(1-(-1))/(max-min). Each attribute is scaled to \[-1;1\] range

### Supported kernels
 * `--linear`: xi \* xj
 * `--polynomial`: (xi\*xj+c)^d
 * `--gaussian`: exp(-gamma * ||xi-xj||^2)

### Supported multiclass classification methods
 * `--1A1` - One against one (trains k\*(k-1)/2 reduced models)
 * `--1AA` - One against all (trains k models)

## psvm4r

### Compilation and installation
Run `R CMD INSTALL psvm4r` from the root directory of the project. This should install the package in your user packages directory.
When editing the source code, it may be necessary to recompile the attributes. Run the following commands in an R prompt from the root directory of this project:
```R
library(Rcpp)
Rcpp::compileAttributes('psvm4r')
```

### Description
There are several S4 classes implemented in psvm4r that bind to libpsvm objects. We can divide them among Kernel and Classifier classes:

#### Kernel classes
These classes support an `object$get(x,y)` method. Where `x` and `y` are numeric vectors. This returns the element in the kernel matrix for vectors `x` and `y`.
The following classes are available with their respective constructor parameters:
 * `LinearKernel()`
 * `PolynomialKernel(degree, constant)`
 * `RbfKernel(gamma)`

#### Classifier classes
These classes support the following methods:
 * `object$train(attributes, labels, C)`. Where `attributes` is a single- or multi-dimensional vector of numeric attributes, `labels` is a single vector of integer labels and `C` is a numeric value for the regularization parameter.
 * `object$predict(attributes)`. Predicts on a trained classifier object.

The following classes are available with their respective constructor parameters:
 * `OneAgainstOneCSVC(num_classes, num_dimensions, K)`
 * `OneAgainstAllCSVC(num_classes, num_dimensions, K)`

### Demos
There are a few demos included, demonstrating the decision boundaries of each kernel. To run them, run one of the following from the R command line:
 * `demo('linear')`
 * `demo('quadratic')`
 * `demo('rbf')`

![linear](https://i.imgur.com/8HEZaOm.png)
![quadratic](https://i.imgur.com/u0h3CiZ.png)
![rbf](https://i.imgur.com/FrWpzFm.png)

## References / Recommended reads:
 * [Buttou, L., Lin, C. (2006). *Support Vector Machine Solvers*.](http://leon.bottou.org/publications/pdf/lin-2006.pdf)
 * [Burges, C. (1998). *A Tutorial on support Vector Machines for Pattern Recognition*.](http://www.di.ens.fr/~mallat/papiers/svmtutorial.pdf)
 * [Chang, C., Lin, C. (2001). *LIBSVM: A Library for Support Vector Machines*.](https://www.csie.ntu.edu.tw/~cjlin/papers/libsvm.pdf)
 * [Berwick, R. *An Idiot's Guide to Support Vector Machines (SVMs).*](http://web.mit.edu/6.034/wwwbob/svm-notes-long-08.pdf)
