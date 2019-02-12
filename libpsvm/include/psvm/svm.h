#pragma once

#include <vector>
#include <cstddef>
#include <memory>

#include "kernels.h"

class SVM {
public:
	SVM(size_t dimensions, std::shared_ptr<const Kernel> kernel);

	size_t getSupportVectorCount() const;
	void fit(const std::vector<double>& x, const std::vector<int>& y, const std::vector<double>& alpha, double C);
	double decision(const double* first) const;

	size_t num_dimensions;
	double bias;
	std::shared_ptr<const Kernel> kernel;
	std::vector<double> sv_alpha_y; // alpha_i * y_i of each sv
	std::vector<double> sv_x;       // x_i of each sv
};
