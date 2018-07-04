# psvm
Sequential and parallel (CUDA) implementations of SVM classifiers.

## Features

Run `./svm --help` or `./cusvm --help` to see a list of parameters.

### Supported input formats
 * CSV
 * IDX (see the MNIST handwritten digit dataset format)

Note that it is required that attributes and labels are in separate files.

### Supported kernels
 * Linear: xi \* xj
 * Polynomial: (xi\*xj+c)^d
 * Gaussian: exp(-||xi-xj||^2/(2\*gamma^2))

### Supported multiclass classification methods
 * 1A1 - One against one (trains k\*(k-1)/2 reduced models)
 * 1AA - One against all (trains k models)
 * TWOCLASS - Traditional two-class classification.

## Compilation
 * Clone this repository
 * Download submodules
   * `$ git submodule init`
   * `$ git submodule update`
 * `$ make`

## References / Recommended reads:
 * [Buttou, L., Lin, C. (2006). *Support Vector Machine Solvers*.](http://leon.bottou.org/publications/pdf/lin-2006.pdf)
 * [Burges, C. (1998). *A Tutorial on support Vector Machines for Pattern Recognition*.](http://www.di.ens.fr/~mallat/papiers/svmtutorial.pdf)
 * [Chang, C., Lin, C. (2001). *LIBSVM: A Library for Support Vector Machines*.](https://www.csie.ntu.edu.tw/~cjlin/papers/libsvm.pdf)
 * [Berwick, R. *An Idiot's Guide to Support Vector Machines (SVMs).*](http://web.mit.edu/6.034/wwwbob/svm-notes-long-08.pdf)
