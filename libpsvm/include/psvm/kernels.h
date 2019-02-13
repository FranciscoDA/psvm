#pragma once

#include <cmath>

#ifdef __CUDACC__
#define CUDA_CALLABLE_MEMBER __host__ __device__
#else
#define CUDA_CALLABLE_MEMBER
#endif

class Kernel {
public:
	CUDA_CALLABLE_MEMBER virtual double operator()(const double*, const double*, const double*) const = 0;
	virtual Kernel* clone() const = 0;
	virtual ~Kernel();
};

class LinearKernel : public Kernel {
public:
	CUDA_CALLABLE_MEMBER double operator()(const double* x1, const double* x2, const double* y1) const override {
		double result = 0.;
		while (x1 != x2) {
			result += (*x1) * (*y1);
			++x1;
			++y1;
		}
		return result;
	}
	LinearKernel* clone() const override;
};

class RbfKernel : public Kernel {
public:
	RbfKernel(double gamma);
	CUDA_CALLABLE_MEMBER double operator()(const double* x1, const double* x2, const double* y1) const override {
		double result = 0.;
		while (x1 != x2) {
			result += (*x1 - *y1) * (*x1 - *y1);
			++x1;
			++y1;
		}
		return exp(-gamma * result);
	}
	const double gamma;
	RbfKernel* clone() const override;
};

class PolynomialKernel : public Kernel {
public:
	PolynomialKernel(double d, double c);
	CUDA_CALLABLE_MEMBER double operator()(const double* x1, const double* x2, const double* y1) const override {
		double result = constant;
		while (x1 != x2) {
			result += (*x1) * (*y1);
			++x1;
			++y1;
		}
		return pow(result, degree);
	}
	const double degree;
	const double constant;
	PolynomialKernel* clone() const override;
};

