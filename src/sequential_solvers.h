
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
			if (y[k] == 1 and alpha[k] < C or y[k] == -1 and alpha[k] > 0) {
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
			if ((y[k] == 1 and alpha[k] > 0 or y[k] == -1 and alpha[k] < C)) {
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

/* modified gradient projection method implementation */
template<typename SVMT>
unsigned int mgp(SVMT& svm, const vector<double>& x, const vector<double>& y, double epsilon, double C) {
	size_t n = y.size();
	size_t d = svm.getD();
	vector<double> alpha(n, 0.0);
	vector<double> g(n, 0.0);
	vector<double> u(n, 0.0);

	vector<int> violating_samples(n);
	for (int i = 0; i < violating_samples.size(); ++i)
	violating_samples[i] = i;

	unsigned int iterations = 0;
	while(violating_samples.size() > 0) {
		++iterations;
		vector<int> working_set = violating_samples;
		while (true) {
			for (int k : working_set) {
				g[k] = 1.0;
				for (int i = 0; i < n; ++i)
				g[k] -= y[k] * y[i] * svm.kernel(&x[k*d], &x[i*d]) * alpha[i];
			}
			while(true) {
				double rho = 0.0;
				for (int i : working_set)
				rho += y[i] * g[i]/double(working_set.size());
				for (int k = 0; k < n; k++) {
					bool in_ws = false;
					for (int i : working_set) {
						if (k == i) {
							in_ws = true;
							break;
						}
					}
					u[k] = in_ws ? g[k] - y[k] * rho : 0.0;
				}

				int old_size = working_set.size();
				for (auto it = begin(working_set); it != end(working_set);)
				if ((u[*it] > 0 and alpha[*it] >= C) or (u[*it] < 0 and alpha[*it] <= 0))
					it = working_set.erase(it);
				else
				++it;
				if (old_size == working_set.size())
				break;
			}
			bool all_null = true;
			for (double u_k : u)
			if (u_k > epsilon) {
				all_null = false;
				break;
			}
			if (all_null)
			break;

			double gu = inner_product(begin(g), end(g), begin(u), 0.0);
			double uhu = 0.0;
			for (int i = 0; i < n; i++)
				for (int j = 0; j < n; j++)
					uhu += u[i] * y[i] * y[j] * svm.kernel(&x[i*d], &x[j*d]) * u[j];
			// lambda_plus is gu/uHu
			double lambda_max = numeric_limits<double>::infinity();
			for (int i = 0; i < n; ++i)
			lambda_max = min(lambda_max, u[i] != 0 ? max(0.0, C / u[i]) - alpha[i] / u[i] : lambda_max);
			double lambda_star = max(0.0, min(lambda_max, gu/uhu));

			for (int i = 0; i < n; i++)
				alpha[i] += lambda_star * u[i];
		}
		violating_samples.clear();
		for (int i = 0; i < n; i++) {
			if (alpha[i] < 0-epsilon || alpha[i] > C+epsilon)
				violating_samples.push_back(i);
		}
	}
	svm.fit(x, y, alpha, C);
	return iterations;
}

#endif
