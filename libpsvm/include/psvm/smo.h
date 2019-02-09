#pragma once
#include <vector>
#include "svm.h"

template <typename KT>
unsigned int smo(SVM<KT>& svm, const std::vector<double>& x, const std::vector<int>& y, double epsilon, double C);

extern template unsigned int smo(SVM<LinearKernel>&, const std::vector<double>&, const std::vector<int>&, double, double);
extern template unsigned int smo(SVM<RbfKernel>&, const std::vector<double>&, const std::vector<int>&, double, double);
extern template unsigned int smo(SVM<PolynomialKernel>&, const std::vector<double>&, const std::vector<int>&, double, double);
