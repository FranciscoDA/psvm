
#ifndef _CUDA_SOLVERS_H_
#define _CUDA_SOLVERS_H_

#include <limits>
#include <numeric>
#include <cmath>
#include <algorithm>
#include <thrust/device_vector.h>

#include "svm.h"

using namespace std;

/* sequential minimal optimization method */
template<typename SVMT>
unsigned int smo(SVMT& svm, const vector<double>& x, const vector<int>& y, double epsilon, double C) {
	size_t n = y.size();
	size_t d = svm.getD();

	typename SVMT::kernel_type* d_kernel;
	cudaMalloc(&d_kernel, sizeof(typename SVMT::kernel_type));
	cudaMemcpy(d_kernel, (void*) &svm.kernel, sizeof(typename SVMT::kernel_type), cudaMemcpyHostToDevice);

	thrust::device_vector<double> d_x = x;
	auto d_x_ptr = d_x.data().get();
	thrust::device_vector<int> d_y = y;
	auto d_y_ptr = d_y.data().get();
	thrust::device_vector<double> d_alpha (n, 0.);
	auto d_alpha_ptr = d_alpha.data().get();
	thrust::device_vector<double> d_g(n, 0.);
	auto d_g_ptr = d_g.data().get();

	thrust::counting_iterator<int> ids_begin(0);
	thrust::counting_iterator<int> ids_end = ids_begin+n;

	thrust::device_vector<double> d_k_cache(n);
	auto d_k_cache_ptr = d_k_cache.data().get();
	thrust::transform(ids_begin, ids_end, d_k_cache.begin(), [d,d_kernel,d_x_ptr] __device__ (int i){
		return d_kernel->operator()(d_x_ptr+i*d, d_x_ptr+i*d, d);
	});
	thrust::device_vector<double> d_ki_cache(n);
	auto d_ki_cache_ptr = d_ki_cache.data().get();

	using search_t = thrust::pair<double, int>;
	auto map_i = [=] __device__ (int i) {
		double B = (C * d_y_ptr[i] + C)/2.0;
		return search_t(d_y_ptr[i] * d_alpha_ptr[i] < B ? d_y_ptr[i] * d_g_ptr[i] : -INFINITY, i);
	};

	auto map_jg = [=] __device__ (int i) {
		double A = (C * d_y_ptr[i] - C)/2.0;
		return A < d_y_ptr[i] * d_alpha_ptr[i] ? d_y_ptr[i] * d_g_ptr[i] : INFINITY;
	};

	unsigned int iterations = 0;
	while(true) {
		++iterations;
		auto i_result = thrust::transform_reduce(ids_begin, ids_end, map_i, search_t(-INFINITY, -1), thrust::maximum<search_t>());
		double g_max = thrust::get<0>(i_result);
		int i = thrust::get<1>(i_result);

		/* fill the cache for the ith row of the kernel matrix */
		thrust::transform(ids_begin, ids_end, d_ki_cache.begin(), [i,d,d_kernel,d_x_ptr] __device__ (int k){
			return (*d_kernel)(d_x_ptr + i*d, d_x_ptr + k*d, d);
		});

		/* get the minimum value for the gradient. note that this is not the second example to optimize */
		double g_min = thrust::transform_reduce(ids_begin, ids_end, map_jg, INFINITY, thrust::minimum<double>());

		auto map_jo = [=] __device__ (int k) {
			double A = (C * d_y_ptr[k] - C)/2.0;
			if (d_y_ptr[i] * d_alpha_ptr[i] < A)
				return search_t(INFINITY, -1);
			double b = g_max + d_y_ptr[k] * d_g_ptr[k];
			double lambda = thrust::max(d_k_cache_ptr[i] + d_k_cache_ptr[k] - 2 * d_ki_cache_ptr[k], 1e-12);
			return b > 0. ? search_t(-(b*b)/lambda, k) : search_t(INFINITY, -1);
		};
		search_t j_result = thrust::transform_reduce(ids_begin, ids_end, map_jo, search_t(INFINITY, -1), thrust::minimum<search_t>());
		int j = thrust::get<1>(j_result);

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

		thrust::transform(ids_begin, ids_end, d_g.begin(), [=] __device__ (int k) {
			double Kik = d_ki_cache_ptr[k];
			double Kjk = (*d_kernel)(d_x_ptr + j*d, d_x_ptr + k*d, d);
			return d_g_ptr[k] + d_y_ptr[k] * (Kik * delta_ai * d_y_ptr[i] + Kjk * delta_aj * d_y_ptr[j]);
		});
	}
	cudaFree(d_kernel);
	vector<double> alpha (d_alpha.begin(), d_alpha.end());
	svm.fit(x, y, alpha, C);

	return iterations;
}

#endif
