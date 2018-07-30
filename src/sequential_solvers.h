
#ifndef _SEQUENTIAL_SOLVERS_H_
#define _SEQUENTIAL_SOLVERS_H_

#include <limits>
#include <numeric>
#include <algorithm>

#include "svm.h"

using namespace std;

/* sequential minimal optimization method implementation */
template<typename SVMT>
unsigned int smo(SVMT& svm, const vector<double>& x, const vector<int>& y, double epsilon, double C) {
	size_t n = y.size();
	size_t d = svm.getD();
	vector<double> alpha(n, 0.0);
	vector<double> g(n, -1.0); // gradient
	vector<double> k_cache(n); // cache for the diagonal of the kernel matrix
	for (int i = 0; i < n; ++i)
		k_cache[i] = svm.kernel(&x[i*d], &x[i*d]);
	vector<double> ki_cache(n); // cache for the ith row of the kernel matrix

	unsigned int iterations = 0;
	while(true) {
		++iterations;

		int i = -1;
		double g_max = -numeric_limits<double>::infinity();
		double g_min = numeric_limits<double>::infinity();
		for (int k = 0; k < n; ++k) {
			if ((y[k] == 1 and alpha[k] < C) or (y[k] == -1 and alpha[k] > 0)) {
				if (-y[k] * g[k] >= g_max) {
					i = k;
					g_max = -y[k] * g[k];
				}
			}
		}
		for (int k = 0; k < n; ++k)
			ki_cache[k] = svm.kernel(&x[i*d], &x[k*d]);

		int j = -1;
		double obj_min = numeric_limits<double>::infinity();
		for (int k = 0; k < n; ++k) {
			if ((y[k] == 1 and alpha[k] > 0) or (y[k] == -1 and alpha[k] < C)) {
				double b = g_max + y[k] * g[k];
				if (-y[k] * g[k] <= g_min) {
					g_min = -y[k]*g[k];
				}
				if (b > 0.) {
					double lambda = k_cache[i] + k_cache[k] - 2 * ki_cache[k];
					lambda = max(lambda, 1e-12);
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

		double lambda = max(Kii + Kjj - 2 * Kij, 1e-12);
		double step = (-y[i] * g[i] + y[j] * g[j])/lambda;

		const double old_ai = alpha[i];
		const double old_aj = alpha[j];

		alpha[i] += y[i] * step;
		alpha[j] -= y[j] * step;

		double sum = y[i] * old_ai + y[j] * old_aj;
		if (alpha[i] < 0.) alpha[i] = 0.;
		if (alpha[i] > C)  alpha[i] = C;
		alpha[j] = y[j] * (sum - y[i] * alpha[i]);
		if (alpha[j] < 0.) alpha[j] = 0.;
		if (alpha[j] > C)  alpha[j] = C;
		alpha[i] = y[i] * (sum - y[j] * alpha[j]);

		const double delta_ai = alpha[i] - old_ai;
		const double delta_aj = alpha[j] - old_aj;
		for (int k = 0; k < n; ++k) {
			const double Kik = ki_cache[k];
			const double Kjk = svm.kernel(&x[j*d], &x[k*d]);
			g[k] += y[k] * (Kik * delta_ai * y[i] + Kjk * delta_aj * y[j]);
		}
	}
	svm.fit(x, y, alpha, C);
	return iterations;
}

#endif
