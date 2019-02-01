#define SUPPRESS_SVM_EXTERN_TEMPLATES
#include "psvm/svm.h"
#undef  SUPPRESS_SVM_EXTERN_TEMPLATES

template class SVM<LinearKernel>;
template class SVM<RbfKernel>;
template class SVM<PolynomialKernel>;
