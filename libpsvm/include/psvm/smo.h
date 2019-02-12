#pragma once
#include <vector>
#include "svm.h"

template <typename KT>
unsigned int smo(SVM& svm, const std::vector<double>& x, const std::vector<int>& y, double epsilon, double C);

extern template unsigned int smo<LinearKernel>(SVM&, const std::vector<double>&, const std::vector<int>&, double, double);
extern template unsigned int smo<PolynomialKernel>(SVM&, const std::vector<double>&, const std::vector<int>&, double, double);
extern template unsigned int smo<RbfKernel>(SVM&, const std::vector<double>&, const std::vector<int>&, double, double);

