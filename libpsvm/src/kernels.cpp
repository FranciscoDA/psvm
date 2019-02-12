#include "psvm/kernels.h"

Kernel::~Kernel() {
}

LinearKernel* LinearKernel::clone() const {
	return new LinearKernel();
}

RbfKernel::RbfKernel(double gamma) : gamma(gamma) {}

RbfKernel* RbfKernel::clone() const {
	return new RbfKernel(gamma);
}


PolynomialKernel::PolynomialKernel(double degree, double constant) : degree(degree), constant(constant) {}

PolynomialKernel* PolynomialKernel::clone() const {
	return new PolynomialKernel(degree, constant);
}




