#define SUPPRESS_EXTERN_TEMPLATES
#include "psvm/sequential_solvers.h"
#undef  SUPPRESS_EXTERN_TEMPLATES

template unsigned int smo(SVM<LinearKernel>&, const std::vector<double>&, const std::vector<int>&, double, double);
template unsigned int smo(SVM<RbfKernel>&, const std::vector<double>&, const std::vector<int>&, double, double);
template unsigned int smo(SVM<PolynomialKernel>&, const std::vector<double>&, const std::vector<int>&, double, double);
