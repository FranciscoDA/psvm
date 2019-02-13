#pragma once

#include <cmath>

#ifdef __CUDACC__
#define CUDA_CALLABLE_MEMBER __host__ __device__
#else
#define CUDA_CALLABLE_MEMBER
#endif

class Kernel {
public:
	virtual double operator()(const double*, const double*, const double*) const = 0;
	virtual Kernel* clone() const = 0;
	virtual ~Kernel();
};

class LinearKernel : public Kernel {
public:
	double operator()(const double* x1, const double* x2, const double* y1) const override;
	LinearKernel* clone() const override;

	static CUDA_CALLABLE_MEMBER double Apply(const LinearKernel* kernel, const double* x1, const double* x2, const double* y1) {
		double result = 0.;
		while (x1 != x2) {
			result += (*x1) * (*y1);
			++x1;
			++y1;
		}
		return result;
	}
};

class RbfKernel : public Kernel {
public:
	RbfKernel(double gamma);
	double operator()(const double* x1, const double* x2, const double* y1) const override;
	const double gamma;
	RbfKernel* clone() const override;

	static CUDA_CALLABLE_MEMBER double Apply(const RbfKernel* kernel, const double* x1, const double* x2, const double* y1) {
		double result = 0.;
		while (x1 != x2) {
			result += (*x1 - *y1) * (*x1 - *y1);
			++x1;
			++y1;
		}
		return exp(-kernel->gamma * result);
	}
};

class PolynomialKernel : public Kernel {
public:
	PolynomialKernel(double d, double c);
	double operator()(const double* x1, const double* x2, const double* y1) const override;
	const double degree;
	const double constant;
	PolynomialKernel* clone() const override;

	static CUDA_CALLABLE_MEMBER double Apply(const PolynomialKernel* kernel, const double* x1, const double* x2, const double* y1) {
		double result = kernel->constant;
		while (x1 != x2) {
			result += (*x1) * (*y1);
			++x1;
			++y1;
		}
		return pow(result, kernel->degree);
	}
};

