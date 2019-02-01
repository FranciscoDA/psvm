#pragma once

#include <algorithm>
#include <limits>
#include <numeric>
#include <algorithm>
#include <vector>
#include <cmath>

#include "svm.h"

/* sequential minimal optimization method implementation */
template<typename KT>
unsigned int smo(SVM<KT>& svm, const std::vector<double>& x, const std::vector<int>& y, double epsilon, double C) {
	size_t n = y.size();
	size_t d = svm.getDimensions();
	std::vector<double> alpha(n, 0.0);
	std::vector<double> g(n, -1.0); // gradient
	std::vector<double> k_cache(n); // cache for the diagonal of the kernel matrix
	for (int i = 0; i < n; ++i)
		k_cache[i] = svm.kernel(&x[i*d], &x[i*d+d], &x[i*d]);
	std::vector<double> ki_cache(n); // cache for the ith row of the kernel matrix

	unsigned int iterations = 0;
	while(true) {
		++iterations;

		int i = -1;
		double g_max = -std::numeric_limits<double>::infinity();
		double g_min = std::numeric_limits<double>::infinity();
		for (int k = 0; k < n; ++k) {
			if (alpha[k] * y[k] < (C * y[k] + C)/2.) {
				if (-y[k] * g[k] >= g_max) {
					i = k;
					g_max = -y[k] * g[k];
				}
			}
		}

		// fill cache for the ith row of the kernel matrix
		for (int k = 0; k < n; ++k)
			ki_cache[k] = svm.kernel(&x[i*d], &x[i*d+d], &x[k*d]);

		int j = -1;
		double obj_min = std::numeric_limits<double>::infinity();
		for (int k = 0; k < n; ++k) {
			if (alpha[k]*y[k] > (C * y[k] - C)/2.) {
				if (-y[k] * g[k] <= g_min) {
					g_min = -y[k]*g[k];
				}
				double b = g_max + y[k] * g[k];
				if (b > 0.) {
					double lambda = k_cache[i] + k_cache[k] - 2 * ki_cache[k];
					lambda = std::max(lambda, 1e-12);
					if (-(b*b)/lambda <= obj_min) {
						j = k;
						obj_min = -(b*b)/lambda;
					}
				}
			}
		}

		if (g_max - g_min < epsilon or i==-1 or j==-1)
			break;

		const double Kii = k_cache[i];
		const double Kjj = k_cache[j];
		const double Kij = ki_cache[j];

		double lambda = std::max(Kii + Kjj - 2 * Kij, 1e-12);
		double step = (-y[i] * g[i] + y[j] * g[j])/lambda;

		const double old_ai = alpha[i];
		const double old_aj = alpha[j];

		alpha[i] += y[i] * step;
		alpha[j] -= y[j] * step;

		double sum = y[i] * old_ai + y[j] * old_aj;
		alpha[i] = std::clamp(alpha[i], 0., C);
		alpha[j] = y[j] * (sum - y[i] * alpha[i]);
		alpha[j] = std::clamp(alpha[j], 0., C);
		alpha[i] = y[i] * (sum - y[j] * alpha[j]);

		const double delta_ai = alpha[i] - old_ai;
		const double delta_aj = alpha[j] - old_aj;
		for (int k = 0; k < n; ++k) {
			const double Kik = ki_cache[k];
			const double Kjk = svm.kernel(&x[j*d], &x[j*d+d], &x[k*d]);
			g[k] += y[k] * (Kik * delta_ai * y[i] + Kjk * delta_aj * y[j]);
		}
	}
	svm.fit(x, y, alpha, C);
	return iterations;
}

#ifndef SUPPRESS_EXTERN_TEMPLATES
extern template unsigned int smo(SVM<LinearKernel>&, const std::vector<double>&, const std::vector<int>&, double, double);
extern template unsigned int smo(SVM<RbfKernel>&, const std::vector<double>&, const std::vector<int>&, double, double);
extern template unsigned int smo(SVM<PolynomialKernel>&, const std::vector<double>&, const std::vector<int>&, double, double);
#endif
