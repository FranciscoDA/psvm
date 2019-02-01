#include "psvm/kernels.h"

RbfKernel::RbfKernel(double gamma) : _gamma(-2*gamma*gamma) {}
PolynomialKernel::PolynomialKernel(double degree, double constant) : _degree(degree), _constant(constant) {}

