#include "psvm/kernels.h"

RbfKernel::RbfKernel(double gamma) : _gamma(-gamma) {}
PolynomialKernel::PolynomialKernel(double degree, double constant) : _degree(degree), _constant(constant) {}

