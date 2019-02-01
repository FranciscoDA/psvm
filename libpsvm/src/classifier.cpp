#define SUPPRESS_CLASSIFIER_EXTERN_TEMPLATES
#include "psvm/classifier.h"
#undef  SUPPRESS_CLASSIFIER_EXTERN_TEMPLATES

template class OneAgainstAllSVC<LinearKernel>;
template class OneAgainstAllSVC<RbfKernel>;
template class OneAgainstAllSVC<PolynomialKernel>;

template class OneAgainstOneSVC<LinearKernel>;
template class OneAgainstOneSVC<RbfKernel>;
template class OneAgainstOneSVC<PolynomialKernel>;
