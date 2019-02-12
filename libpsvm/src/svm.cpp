#include "psvm/svm.h"

SVM::SVM(size_t dimensions, std::shared_ptr<const Kernel> kernel) : num_dimensions(dimensions), bias(0.0), kernel(kernel) {
}

size_t SVM::getSupportVectorCount() const {
	return sv_alpha_y.size();
}

void SVM::fit(const std::vector<double>& x, const std::vector<int>& y, const std::vector<double>& alpha, double C) {
	sv_x.clear();
	sv_alpha_y.clear();
	for (size_t i = 0; i < y.size(); ++i) {
		if (alpha[i] > 0.) {
			sv_alpha_y.push_back(y[i] * alpha[i]);
			sv_x.insert(end(sv_x), begin(x) + i * num_dimensions, begin(x) + (i+1) * num_dimensions);
		}
	}
	// calculate b from a support vector that lies on a margin (ie: 0 < alpha_i < C)
	bias = 0.;
	double b_sum = 0.;
	double b_count = 0.;
	for (size_t i = 0; i < sv_alpha_y.size(); ++i) {
		double Ai = sv_alpha_y[i] > 0. ? 0. : -C;
		double Bi = sv_alpha_y[i] > 0. ? C  : 0.;
		if (Ai < sv_alpha_y[i] && sv_alpha_y[i] < Bi) {
			// sum alpha_i y_i K(x_j,sv_i) + b = y_i
			// therefore:
			// y_i - sum alpha_i y_i K(x_j, sv_i) = b
			b_sum += (sv_alpha_y[i] > 0. ? 1. : -1.) - decision(&sv_x[i * num_dimensions]);
			b_count += 1.0;
		}
	}
	bias = b_sum/b_count; // b is averaged among the support vectors
}

double SVM::decision(const double* first) const {
	double sum = bias;
	for (size_t i = 0; i < sv_alpha_y.size(); ++i) {
		sum += sv_alpha_y[i] * kernel->operator()(first, first + num_dimensions, &sv_x[i * num_dimensions]);
	}
	return sum;
}

