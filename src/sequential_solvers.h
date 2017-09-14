
#ifndef _SEQUENTIAL_SOLVERS_H_
#define _SEQUENTIAL_SOLVERS_H_

#include <limits>
#include <numeric>
#include <algorithm>

#include "svm.h"

using namespace std;

/* sequential minimal optimization method implementation */
template<typename SVMT>
unsigned int smo(SVMT& svm, const vector<double>& x, const vector<double>& y, double epsilon, double C) {
	size_t n = y.size();
	size_t d = svm.getD();
	vector<double> alpha(n, 0.0);
	vector<double> g(n, 1.0);

	unsigned int iterations = 0;
	while(true) {
		++iterations;
		int i = 0, j = 0;
		double i_max = -numeric_limits<double>::infinity();
		double j_min = numeric_limits<double>::infinity();

		for (int k = n-1; k >= 0; k--) {
			double A = (C * y[k] - C)/2.0;
			double B = (C * y[k] + C)/2.0;
			if (y[k] * alpha[k] < B and i_max < y[k] * g[k]) {
				i = k;
				i_max = y[k] * g[k];
			}
			if (A < y[k] * alpha[k] and y[k] * g[k] < j_min) {
				j = k;
				j_min = y[k] * g[k];
			}
		}
		//std::cout << "i: " << i << " i_max: " << i_max << " j: " << j << " j_min: " << j_min << endl;

		if (i_max - j_min < epsilon) break;

		double Kii = svm.kernel(&x[i*d], &x[i*d]),
		Kij = svm.kernel(&x[i*d], &x[j*d]),
		Kjj = svm.kernel(&x[j*d], &x[j*d]);

		double Aj = (C * y[j] - C)/2.0;
		double Bi = (C * y[i] + C)/2.0;
		double lambda = min(Bi - y[i] * alpha[i], y[j] * alpha[j] - Aj);
		lambda = min(lambda, (i_max-j_min)/(Kii+Kjj-2*Kij));

		for(int k = 0; k < n; k++) {
			double Kik = svm.kernel(&x[i*d], &x[k*d]),
			Kjk = svm.kernel(&x[j*d], &x[k*d]);
			g[k] += lambda * y[k] * (Kjk - Kik);
		}
		alpha[i] += y[i] * lambda;
		alpha[j] -= y[j] * lambda;
	}
	svm.fit(x, y, alpha, epsilon, C);
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
	svm.fit(x, y, alpha, epsilon, C);
	return iterations;
}

#endif
