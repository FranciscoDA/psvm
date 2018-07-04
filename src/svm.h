
#ifndef _SVM_H_
#define _SVM_H_

#ifdef __CUDACC__
#define CUDA_CALLABLE_MEMBER __host__ __device__
#else
#define CUDA_CALLABLE_MEMBER
#endif

#include <vector>
#include <cmath>

using namespace std;

enum class KERNEL_TYPE {
	LINEAR,
	RBF,
	POLYNOMIAL
};

template<KERNEL_TYPE KT>
class Kernel {
};

template<>
class Kernel<KERNEL_TYPE::LINEAR> {
public:
	CUDA_CALLABLE_MEMBER double K(const double* x1, const double* x2, size_t d) const {
		double result = 0.0;
		for (size_t i = 0; i < d; ++i)
			result += x1[i] * x2[i];
		return result;
	}
};

template<>
class Kernel<KERNEL_TYPE::RBF> {
public:
	Kernel(double gamma) : _gamma(-gamma*gamma) {
	}
	CUDA_CALLABLE_MEMBER double K(const double* x1, const double* x2, size_t d) const {
		double result = 0.0;
		for (size_t i = 0; i < d; ++i)
			result += (x1[i] - x2[i]) * (x1[i] - x2[i]);
		return exp(_gamma * result);
	}
private:
	const double _gamma;
};

template<>
class Kernel<KERNEL_TYPE::POLYNOMIAL> {
public:
	Kernel(double d, double c) : _d(d), _c(c) {
	}
	CUDA_CALLABLE_MEMBER double K(const double* x1, const double* x2, size_t d) const {
		double result = _c;
		for (size_t i = 0; i < d; ++i)
			result += x1[i] * x2[i];
		return pow(result, _d);
	}
private:
	const double _d;
	const double _c;
};

template<KERNEL_TYPE KT>
class SVM {
public:
	SVM(size_t d, const Kernel<KT>& k) : _d(d), _kernel(k), _b(0.0) {
	}

	size_t getD() const {
		return _d;
	}
	double getBias() const {
		return _b;
	}
	void fit(const std::vector<double>& x, const std::vector<int>& y, const std::vector<double>& alpha, double epsilon, double C) {
		_sv_x.clear();
		_sv_alpha_y.clear();
		for (int i = 0; i < y.size(); ++i) {
			if (alpha[i] > epsilon) {
				_sv_alpha_y.push_back(y[i] * alpha[i]);
				_sv_x.insert(end(_sv_x), begin(x) + i*_d, begin(x) + (i+1)*_d);
			}
		}
		// calculate b from a support vector that lies on a margin (ie: 0 < alpha_i < C)
		double b_sum = 0.0;
		double b_count = 0.0;
		for (int i = 0; i < _sv_alpha_y.size(); ++i) {
			if (epsilon < _sv_alpha_y[i] && _sv_alpha_y[i] < C-epsilon) {
				// sum alpha_i y_i K(x_j,sv_i) + b = 1
				// 1 - sum alpha_i y_i K(x_j, sv_i) = b
				b_sum += 1.0 - decision(&_sv_x[i*_d]);
				b_count += 1.0;
			}
			else if (-C+epsilon < _sv_alpha_y[i] && _sv_alpha_y[i] < -epsilon) {
				// -1 - sum alpha_i y_i K(x_j, sv_i) = b
				b_sum += -1.0 - decision(&_sv_x[i*_d]);
				b_count += 1.0;
			}
		}
		_b = b_sum/b_count; // b is averaged among the support vectors
	}
	double decision(const double* x) const {
		double sum = _b;
		for (int i = 0; i < _sv_alpha_y.size(); ++i) {
			sum += _sv_alpha_y[i] * kernel(&_sv_x[i*_d], x);
		}
		return sum;
	}

	bool predict(const double* x) const {
		return decision(x) > 0;
	}

	CUDA_CALLABLE_MEMBER double kernel(const double* x1, const double* x2) const {
		return _kernel.K(x1, x2, _d);
	}

	const std::vector<double>& getSVX() const {
		return _sv_x;
	}
	const std::vector<double>& getSVAlphaY() const {
		return _sv_alpha_y;
	}
	size_t getSupportVectorCount() const {
		return _sv_alpha_y.size();
	}
private:
	size_t _d;
	double _b;
	std::vector<double> _sv_alpha_y; // alpha_i * y_i of each sv
	std::vector<double> _sv_x;       // x_i of each sv
	const Kernel<KT> _kernel;
};

template<typename SVMT>
double decision(const SVMT& svm, const double* x);


#endif
