#define SUPPRESS_CUDA_SOLVERS_EXTERN_TEMPLATES
#include "psvm/cuda_solvers.h"
#undef  SUPPRESS_CUDA_SOLVERS_EXTERN_TEMPLATES

template unsigned int smo(SVM<LinearKernel>&, const std::vector<double>&, const std::vector<int>&, double, double);
template unsigned int smo(SVM<RbfKernel>&, const std::vector<double>&, const std::vector<int>&, double, double);
template unsigned int smo(SVM<PolynomialKernel>&, const std::vector<double>&, const std::vector<int>&, double, double);
