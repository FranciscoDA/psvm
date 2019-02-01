#pragma once

#include <limits>
#include <numeric>
#include <cmath>
#include <algorithm>
#include <thrust/device_vector.h>

#include "svm.h"

/* sequential minimal optimization method */
template<typename KT>
unsigned int smo(SVM<KT>& svm, const std::vector<double>& x, const std::vector<int>& y, double epsilon, double C) {
	size_t n = y.size();
	size_t d = svm.getDimensions();

	KT* d_kernel;
	cudaMalloc(&d_kernel, sizeof(KT));
	cudaMemcpy(d_kernel, (void*) &svm.kernel, sizeof(KT), cudaMemcpyHostToDevice);

	thrust::device_vector<double> d_x = x;
	thrust::device_vector<int>    d_y = y;
	thrust::device_vector<double> d_alpha (n, 0.);
	thrust::device_vector<double> d_g(n, 0.);

	double* dp_x     = d_x.data().get();
	int*    dp_y     = d_y.data().get();
	double* dp_alpha = d_alpha.data().get();
	double* dp_g     = d_g.data().get();

	thrust::counting_iterator<int> ids_first(0);
	thrust::counting_iterator<int> ids_last = ids_first+n;

	auto apply_kernel = [d, d_kernel, dp_x] __device__ (int i, int j) {
		return (*d_kernel)(dp_x + i*d, dp_x + i*d + d, dp_x + j*d);
	};

	// cache for diagonal of kernel matrix
	thrust::device_vector<double> d_k_cache(n);
	thrust::transform(ids_first, ids_last, ids_first, d_k_cache.begin(), apply_kernel);

	// cache for the ith row of kernel matrix
	thrust::device_vector<double> d_ki_cache(n);

	double* dp_k_cache  = d_k_cache.data().get();
	double* dp_ki_cache = d_ki_cache.data().get();

	using search_t = thrust::pair<double, int>;

	auto transform_i = [C, dp_y, dp_alpha, dp_g] __host__ __device__ (int i) {
		const double B = (C * dp_y[i] + C)/2.0;
		return search_t(dp_y[i] * dp_alpha[i] < B ? -dp_y[i]*dp_g[i] : -INFINITY, i);
	};

	auto transform_jg = [C, dp_y, dp_alpha, dp_g] __device__ (int i) {
		const double A = (C * dp_y[i] - C)/2.0;
		return A < dp_y[i] * dp_alpha[i] ? -dp_y[i] * dp_g[i] : INFINITY;
	};

	unsigned int iterations = 0;
	while(true) {
		++iterations;
		auto i_result = thrust::transform_reduce(ids_first, ids_last, transform_i, search_t(-INFINITY, -1), thrust::maximum<search_t>());
		double g_max = i_result.first;
		int i = i_result.second;

		/* fill the cache for the ith row of the kernel matrix */
		thrust::transform(ids_first, ids_last, thrust::constant_iterator<int>(i), d_ki_cache.begin(), apply_kernel);

		/* get the minimum value for the gradient. note that this is not the second example to optimize */
		double g_min = thrust::transform_reduce(ids_first, ids_last, transform_jg, INFINITY, thrust::minimum<double>());

		auto transform_jo = [C, g_max, i, dp_y, dp_alpha, dp_g, dp_k_cache, dp_ki_cache] __device__ (int k) {
			const double A = (C * dp_y[k] - C)/2.0;
			double b = g_max + dp_y[k] * dp_g[k];
			double lambda = max(dp_k_cache[i] + dp_k_cache[k] - 2 * dp_ki_cache[k], 1e-12);
			return search_t((A < dp_y[i] * dp_alpha[i] and b > 0.) ? -(b*b)/lambda : INFINITY, k);
		};

		int j = thrust::transform_reduce(ids_first, ids_last, transform_jo, search_t(INFINITY, -1), thrust::minimum<search_t>()).second;

		if (g_max-g_min < epsilon or i==-1 or j==-1)
			break;

		const double Kii = d_k_cache[i];
		const double Kjj = d_k_cache[j];
		const double Kij = d_ki_cache[j];

		double lambda = max(Kii + Kjj - 2 * Kij, 1e-12);
		double step = (-y[i] * d_g[i] + y[j] * d_g[j])/lambda;

		const double old_ai = d_alpha[i];
		const double old_aj = d_alpha[j];
		double ai = old_ai;
		double aj = old_aj;

		ai += y[i] * step;
		aj -= y[j] * step;

		double sum = y[i] * old_ai + y[j] * old_aj;
		ai = ai < 0. ? 0. : ai > C ? C : ai;
		aj = y[j] * (sum - y[i] * ai);
		aj = aj < 0. ? 0. : aj > C ? C : aj;
		ai = y[i] * (sum - y[j] * aj);

		const double delta_ai = ai - old_ai;
		const double delta_aj = aj - old_aj;
		d_alpha[i] = ai;
		d_alpha[j] = aj;

		thrust::transform(ids_first, ids_last, d_g.begin(),
			[i, j, delta_ai, delta_aj, apply_kernel, dp_ki_cache, dp_g, dp_y] __device__ (int k) {
				double Kik = dp_ki_cache[k];
				double Kjk = apply_kernel(j, k);
				return dp_g[k] + dp_y[k] * (Kik * delta_ai * dp_y[i] + Kjk * delta_aj * dp_y[j]);
			}
		);
	}
	cudaFree(d_kernel);
	std::vector<double> alpha (d_alpha.begin(), d_alpha.end());
	svm.fit(x, y, alpha, C);

	return iterations;
}

#ifndef SUPPRESS_CUDA_SOLVERS_EXTERN_TEMPLATES
extern template unsigned int smo(SVM<LinearKernel>&, const std::vector<double>&, const std::vector<int>&, double, double);
extern template unsigned int smo(SVM<RbfKernel>&, const std::vector<double>&, const std::vector<int>&, double, double);
extern template unsigned int smo(SVM<PolynomialKernel>&, const std::vector<double>&, const std::vector<int>&, double, double);
#endif
