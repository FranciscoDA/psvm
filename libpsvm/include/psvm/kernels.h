#pragma once

#include <cmath>
#ifdef __CUDACC__
#define CUDA_CALLABLE_MEMBER __host__ __device__
#else
#define CUDA_CALLABLE_MEMBER
#endif

class LinearKernel {
public:
	CUDA_CALLABLE_MEMBER double operator()(const double* x1, const double* x2, const double* y1) const {
		double result = 0.;
		while (x1 != x2) {
			result += (*x1) * (*y1);
			++x1;
			++y1;
		}
		return result;
	}
};

class RbfKernel {
public:
	RbfKernel(double gamma);
	CUDA_CALLABLE_MEMBER double operator()(const double* x1, const double* x2, const double* y1) const {
		double result = 0.;
		while (x1 != x2) {
			result += (*x1 - *y1) * (*x1 - *y1);
			++x1;
			++y1;
		}
		return exp(_gamma * result);
	}
private:
	const double _gamma;
};

class PolynomialKernel {
public:
	PolynomialKernel(double d, double c);
	CUDA_CALLABLE_MEMBER double operator()(const double* x1, const double* x2, const double* y1) const {
		double result = _constant;
		while (x1 != x2) {
			result += (*x1) * (*y1);
			++x1;
			++y1;
		}
		return pow(result, _degree);
	}
private:
	const double _degree;
	const double _constant;
};

