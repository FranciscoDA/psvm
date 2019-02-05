#pragma once

#include <vector>
#include <cmath>

#include "kernels.h"

template<typename KT>
class SVM {
public:
	using kernel_type = KT;
	SVM(size_t dimensions, const KT& k) : kernel(k), _dimensions(dimensions), _bias(0.0)  {
	}

	size_t getDimensions() const { return _dimensions; }
	double getBias() const { return _bias; }
	const std::vector<double>& getSVX() const { return _sv_x; }
	const std::vector<double>& getSVAlphaY() const { return _sv_alpha_y; }
	size_t getSupportVectorCount() const { return _sv_alpha_y.size(); }

	void fit(const std::vector<double>& x, const std::vector<int>& y, const std::vector<double>& alpha, double C) {
		_sv_x.clear();
		_sv_alpha_y.clear();
		for (size_t i = 0; i < y.size(); ++i) {
			if (alpha[i] > 0.) {
				_sv_alpha_y.push_back(y[i] * alpha[i]);
				_sv_x.insert(end(_sv_x), begin(x) + i*_dimensions, begin(x) + (i+1)*_dimensions);
			}
		}
		// calculate b from a support vector that lies on a margin (ie: 0 < alpha_i < C)
		_bias = 0.;
		double b_sum = 0.;
		double b_count = 0.;
		for (size_t i = 0; i < _sv_alpha_y.size(); ++i) {
			double Ai = _sv_alpha_y[i] > 0. ? 0. : -C;
			double Bi = _sv_alpha_y[i] > 0. ? C  : 0.;
			if (Ai < _sv_alpha_y[i] && _sv_alpha_y[i] < Bi) {
				// sum alpha_i y_i K(x_j,sv_i) + b = y_i
				// y_i - sum alpha_i y_i K(x_j, sv_i) = b
				b_sum += (_sv_alpha_y[i] > 0. ? 1. : -1.) - decision(&_sv_x[i*_dimensions]);
				b_count += 1.0;
			}
		}
		_bias = b_sum/b_count; // b is averaged among the support vectors
	}

	double decision(const double* first) const {
		double sum = _bias;
		for (size_t i = 0; i < _sv_alpha_y.size(); ++i) {
			sum += _sv_alpha_y[i] * kernel(first, first + _dimensions, &_sv_x[i * _dimensions]);
		}
		return sum;
	}

	const KT kernel;
private:
	size_t _dimensions;
	double _bias;
	std::vector<double> _sv_alpha_y; // alpha_i * y_i of each sv
	std::vector<double> _sv_x;       // x_i of each sv
};

#ifndef SUPPRESS_SVM_EXTERN_TEMPLATES
extern template class SVM<LinearKernel>;
extern template class SVM<RbfKernel>;
extern template class SVM<PolynomialKernel>;
#endif
