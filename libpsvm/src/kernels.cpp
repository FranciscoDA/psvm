#include "psvm/kernels.h"

Kernel::~Kernel() {
}

LinearKernel* LinearKernel::clone() const {
	return new LinearKernel();
}
double LinearKernel::operator()(const double* x1, const double* x2, const double* y1) const {
	return LinearKernel::Apply(this, x1, x2, y1);
}

RbfKernel::RbfKernel(double gamma) : gamma(gamma) {}

RbfKernel* RbfKernel::clone() const {
	return new RbfKernel(gamma);
}
double RbfKernel::operator()(const double* x1, const double* x2, const double* y1) const {
	return RbfKernel::Apply(this, x1, x2, y1);
}


PolynomialKernel::PolynomialKernel(double degree, double constant) : degree(degree), constant(constant) {}

PolynomialKernel* PolynomialKernel::clone() const {
	return new PolynomialKernel(degree, constant);
}
double PolynomialKernel::operator()(const double* x1, const double* x2, const double* y1) const {
	return PolynomialKernel::Apply(this, x1, x2, y1);
}



