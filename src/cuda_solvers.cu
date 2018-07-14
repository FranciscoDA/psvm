
#ifndef _CUDA_SOLVERS_H_
#define _CUDA_SOLVERS_H_

#include <limits>
#include <numeric>
#include <cmath>
#include <algorithm>

#include "svm.h"

#define GRID_SIZE  512
#define BLOCK_SIZE 256

using namespace std;

struct Search_t {
	double score;
	int index;
};

template<typename T>
__global__ void cu_set(int n, T* dst, T value) {
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n)
	dst[i] = value;
}

template<typename F, typename H>
__global__ void cu_map(int n, H* dst, F mapping) {
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n)
	dst[i] = mapping(i);
}

/* optimized map-reduce kernel */
template<typename F, typename G, typename H>
__global__ void cu_mr(off_t offset, H* g_out, F mapping, G reduction, bool reduce_output) {
	__shared__ H sdata[BLOCK_SIZE];
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x * blockDim.x * 2 + threadIdx.x + offset;

	sdata[tid] = reduction(mapping(i), mapping(i+blockDim.x));
	__syncthreads();

	for (unsigned int s = blockDim.x/2; s > warpSize; s >>= 1) {
		if (tid < s)
			sdata[tid] = reduction(sdata[tid], sdata[tid+s]);
		__syncthreads();
	}
	/*
	do not sync threads in the last warp, warps are simd synchronous
	unroll loop. assume warp_size=32
	*/
	if (tid < warpSize)
		for (unsigned int s = warpSize; s > 0; s >>= 1)
			sdata[tid] = reduction(sdata[tid], sdata[tid+s]);

	if (tid == 0)
		g_out[blockIdx.x] = reduce_output ? reduction(sdata[0], g_out[blockIdx.x]) : sdata[0];
}
/* map-reduce for trailing data */
template<typename F, typename G, typename H>
__global__ void cu_mr_tail(size_t offset, size_t n, H* g_out, F mapping, G reduction, bool reduce_output) {
	__shared__ H sdata[BLOCK_SIZE];
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x * blockDim.x * 2 + threadIdx.x + offset;

	if (i >= n) return;
	sdata[tid] = (i+blockDim.x < n) ? reduction(mapping(i), mapping(i+blockDim.x)) : mapping(i);
	__syncthreads();

	for (unsigned int s = blockDim.x/2; s > warpSize; s >>= 1) {
		if (tid < s and i+s < n)
			sdata[tid] = reduction(sdata[tid], sdata[tid+s]);
		__syncthreads();
	}

	if (tid < warpSize)
		for (unsigned int s = warpSize; s > 0; s >>= 1)
			if (i+s < n)
				sdata[tid] = reduction(sdata[tid], sdata[tid+s]);

	if (tid == 0)
		g_out[blockIdx.x] = reduce_output ? reduction(sdata[0], g_out[blockIdx.x]) : sdata[0];
}
template <typename F, typename G, typename H>
H cu_mr_wrapper(size_t n, H* d_buf, F mapping, G reduction) {
	H h_buf[GRID_SIZE];
	const size_t kernel_capacity = BLOCK_SIZE*GRID_SIZE*2;
	size_t i = 0;
	for (; i + kernel_capacity < n; i+= kernel_capacity) {
		cu_mr<<<GRID_SIZE, BLOCK_SIZE>>>(i, d_buf, mapping, reduction, i > 0);
	}
	if (i < n) {
		cu_mr_tail<<<GRID_SIZE, BLOCK_SIZE>>>(i, n, d_buf, mapping, reduction, i > 0);
	}
	cudaMemcpy(h_buf, d_buf, sizeof(H) * GRID_SIZE, cudaMemcpyDeviceToHost);
	H result = h_buf[0];

	int a = n/(BLOCK_SIZE*2) + (n%(BLOCK_SIZE*2) ? 1 : 0);
	for (int j = 1; j < min(GRID_SIZE, a); ++j)
		result = reduction(result, h_buf[j]);
	return result;
}

template<typename SVMT>
__global__ void cusmo_update_gradient(size_t n, size_t d, SVMT* g_svm, double* g_x, int* g_y, double* g_alpha, double* g_g, double lambda, int i, int j)
{
	unsigned int k = blockIdx.x * blockDim.x + threadIdx.x;
	if (k >= n)
	return;

	double Kik = g_svm->kernel(&g_x[i*d], &g_x[k*d]),
	Kjk = g_svm->kernel(&g_x[j*d], &g_x[k*d]);
	g_g[k] += lambda * g_y[k] * (Kjk - Kik);
}
__global__ void cusmo_update_alpha(int* g_y, double* g_alpha, int i, int j, double lambda) {
	g_alpha[i] += g_y[i] * lambda;
	g_alpha[j] -= g_y[j] * lambda;
}

/* sequential minimal optimization method */
template<typename SVMT>
unsigned int smo(SVMT& svm, const vector<double>& x, const vector<int>& y, double epsilon, double C) {
	size_t n = y.size();
	size_t d = svm.getD();

	std::vector<double> alpha (n, 0.0);

	SVMT* d_svm;
	double* d_x;
	int* d_y;
	double* d_alpha;
	double* d_g;
	cudaMalloc(&d_svm,    sizeof(SVMT));
	cudaMalloc(&d_x,      sizeof(double) * x.size());
	cudaMalloc(&d_y,      sizeof(int) * y.size());
	cudaMalloc(&d_alpha,  sizeof(double) * n);
	cudaMalloc(&d_g,      sizeof(double) * n);

	cudaMemcpy(d_svm, (void*) &svm,     sizeof(SVMT),            cudaMemcpyHostToDevice);
	cudaMemcpy(d_x,   (void*) x.data(), sizeof(double)*x.size(), cudaMemcpyHostToDevice);
	cudaMemcpy(d_y,   (void*) y.data(), sizeof(int)*y.size(),    cudaMemcpyHostToDevice);
	cu_set<<<GRID_SIZE, BLOCK_SIZE>>>(n, d_alpha, 0.0);
	cu_set<<<GRID_SIZE, BLOCK_SIZE>>>(n, d_g,     1.0);

	auto map_i = [=] __device__ (int i) {
		double B = (C * d_y[i] + C)/2.0;
		return Search_t {d_y[i] * d_alpha[i] < B ? d_y[i] * d_g[i] : -INFINITY, i};
	};
	auto reduce_i = [] __device__ __host__ (const Search_t& arg1, const Search_t& arg2) {
		return arg1.score < arg2.score ? arg2 : arg1;
	};

	auto map_j = [=] __device__ (int i) {
		double A = (C * d_y[i] - C)/2.0;
		return Search_t {A < d_y[i] * d_alpha[i] ? d_y[i] * d_g[i] : INFINITY, i};
	};
	auto reduce_j = [] __device__ __host__ (const Search_t& arg1, const Search_t& arg2) {
		return arg1.score < arg2.score ? arg1 : arg2;
	};

	Search_t* d_result;
	cudaMalloc(&d_result, sizeof(Search_t)*GRID_SIZE);
	Search_t* d_result2;
	cudaMalloc(&d_result2, sizeof(Search_t)*GRID_SIZE);
	//vector<Search_t> gather_result (GRID_SIZE);
	unsigned int iterations = 0;
	while(true) {
		++iterations;
		Search_t result_i = cu_mr_wrapper(n, d_result, map_i, reduce_i);
		Search_t result_j = cu_mr_wrapper(n, d_result2, map_j, reduce_j);

		int i = result_i.index;
		int j = result_j.index;
		double i_max = result_i.score;
		double j_min = result_j.score;

		//std::cout << "i: " << i << " i_max: " << i_max << " j: " << j << " j_min: " << j_min << endl;

		if (i_max - j_min < epsilon) break;

		double Kii = svm.kernel(&x[i*d], &x[i*d]),
		Kij = svm.kernel(&x[i*d], &x[j*d]),
		Kjj = svm.kernel(&x[j*d], &x[j*d]);

		double Aj = (C * y[j] - C)/2.0;
		double Bi = (C * y[i] + C)/2.0;
		double lambda = min(Bi - y[i] * alpha[i], y[j] * alpha[j] - Aj);
		lambda = min(lambda, (i_max-j_min)/(Kii+Kjj-2*Kij));

		cusmo_update_gradient<<<GRID_SIZE, BLOCK_SIZE>>>(n, d, d_svm, d_x, d_y, d_alpha, d_g, lambda, i, j);
		cusmo_update_alpha<<<1,1>>>(d_y, d_alpha, i, j, lambda);
		alpha[i] += y[i] * lambda;
		alpha[j] -= y[j] * lambda;
	}
	svm.fit(x, y, alpha, C);
	cudaFree(d_result);
	cudaFree(d_result2);
	cudaFree(d_svm);
	cudaFree(d_x);
	cudaFree(d_y);
	cudaFree(d_alpha);
	cudaFree(d_g);
	return iterations;
}

/* modified gradient projection method implementation */
template<typename SVMT>
void mgp(SVMT& svm, const vector<double>& x, const vector<double>& y, double epsilon, double C) {

}

#endif
