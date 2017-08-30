
#ifndef _CUDA_SOLVERS_H_
#define _CUDA_SOLVERS_H_

#include <iostream>
#include <limits>
#include <numeric>

#include "svm.h"

#define NUM_BLOCKS 128
#define BLOCK_SIZE 128

using namespace std;

struct Search_t {
  double score;
  int index;
};

template<typename T>
__global__ void cu_init(int n, T val, T* dst)
{
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    dst[i] = val;
}

/* map-reduce kernel */
template<typename F, typename G, typename H>
__global__ void cu_mr(size_t n, H* g_out, F mapping, G reduction)
{
  __shared__ H sdata[BLOCK_SIZE];

  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int right = blockDim.x < n - blockIdx.x*blockDim.x ? blockDim.x : n - blockIdx.x*blockDim.x;

  if (i >= n)
    return;

  sdata[tid] = mapping(i);
  __syncthreads();

  for (unsigned int s = 1; s < blockDim.x; s *= 2)
  {
    int index = s * 2 * tid;
    if (index+s < right)
    {
      sdata[index] = reduction(sdata[index], sdata[index+s]);
    }
    __syncthreads();
  }
  if (tid == 0) {
    g_out[blockIdx.x] = sdata[0];
  }
}
template<typename SVMT>
__global__ void cusmo_update_gradient(size_t n, size_t d, SVMT* g_svm, double* g_x, double* g_y, double* g_alpha, double* g_g, double lambda, int i, int j)
{
  unsigned int k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k >= n)
    return;

  double Kik = g_svm->kernel(&g_x[i*d], &g_x[k*d]),
         Kjk = g_svm->kernel(&g_x[j*d], &g_x[k*d]);
  g_g[k] += lambda * g_y[k] * (Kjk - Kik);
}
__global__ void cusmo_update_alpha(double* g_y, double* g_alpha, int i, int j, double lambda) {
  g_alpha[i] += g_y[i] * lambda;
  g_alpha[j] -= g_y[j] * lambda;
}

/* sequential minimal optimization method */
template<typename SVMT>
void smo(SVMT& svm, const vector<double>& x, const vector<double>& y) {
  size_t n = y.size();
  size_t d = x.size() / y.size();

  std::vector<double> alpha (n, 0.0);

  SVMT* d_svm;
  double* d_x;
  double* d_y;
  double* d_alpha;
  double* d_g;
  cudaMalloc(&d_svm,    sizeof(SVMT));
  cudaMalloc(&d_x,      sizeof(double) * n * d);
  cudaMalloc(&d_y,      sizeof(double) * n);
  cudaMalloc(&d_alpha,  sizeof(double) * n);
  cudaMalloc(&d_g,      sizeof(double) * n);

  cudaMemcpy(d_svm, (void*) &svm,     sizeof(SVMT),       cudaMemcpyHostToDevice);
  cudaMemcpy(d_x,   (void*) x.data(), sizeof(double)*n*d, cudaMemcpyHostToDevice);
  cudaMemcpy(d_y,   (void*) y.data(), sizeof(double)*n,   cudaMemcpyHostToDevice);
  cu_init<<<NUM_BLOCKS, BLOCK_SIZE>>>(n, 0.0, d_alpha);
  cu_init<<<NUM_BLOCKS, BLOCK_SIZE>>>(n, 1.0, d_g);

  double infinity = numeric_limits<double>::infinity();
  auto map_i = [d_svm, d_x, d_y, d_alpha, d_g, infinity] __device__ (int i) {
    return Search_t {d_y[i] * d_alpha[i] < d_svm->B(d_y[i]) ? d_y[i] * d_g[i] : -infinity, i};
  };
  auto reduce_i = [] __device__ __host__ (const Search_t& arg1, const Search_t& arg2) {
    return arg1.score > arg2.score ? arg1 : arg2;
  };

  auto map_j = [d_svm, d_x, d_y, d_alpha, d_g, infinity] __device__ (int i) {
    return Search_t {d_svm->A(d_y[i]) < d_y[i] * d_alpha[i] ? d_y[i] * d_g[i] : infinity, i};
  };
  auto reduce_j = [] __device__ __host__ (const Search_t& arg1, const Search_t& arg2) {
    return arg1.score < arg2.score ? arg1 : arg2;
  };

  Search_t* d_result;
  cudaMalloc(&d_result, sizeof(Search_t)*NUM_BLOCKS);
  array<Search_t, NUM_BLOCKS> gather_result;
  while(true) {
    cu_mr<<<NUM_BLOCKS, BLOCK_SIZE>>>(n, d_result, map_i, reduce_i);
    cudaMemcpy(gather_result.data(), d_result, sizeof(Search_t) * gather_result.size(), cudaMemcpyDeviceToHost);
    auto result_i = gather_result[0];
    for (int z = 1; z < n/NUM_BLOCKS + n%NUM_BLOCKS ? 0 : 1; ++z)
      result_i = reduce_i(result_i, gather_result[z]);

    cu_mr<<<NUM_BLOCKS, BLOCK_SIZE>>>(n, d_result, map_j, reduce_j);
    cudaMemcpy(gather_result.data(), d_result, sizeof(Search_t) * gather_result.size(), cudaMemcpyDeviceToHost);
    auto result_j = gather_result[0];
    for (int z = 1; z < n/NUM_BLOCKS + n%NUM_BLOCKS ? 0 : 1; ++z)
      result_j = reduce_j(result_j, gather_result[z]);

    int i = result_i.index;
    double i_max = result_i.score;
    int j = result_j.index;
    double j_min = result_j.score;

    if (i_max <= j_min)
      break;

    double Kii = svm.kernel(&x[i*d], &x[i*d]),
           Kij = svm.kernel(&x[i*d], &x[j*d]),
           Kjj = svm.kernel(&x[j*d], &x[j*d]);

    double lambda = min(svm.B(y[i]) - y[i] * alpha[i], y[j] * alpha[j] - svm.A(y[j]));
    lambda = min(lambda, (i_max-j_min)/(Kii+Kjj-2*Kij));

    cusmo_update_gradient<<<NUM_BLOCKS, BLOCK_SIZE>>>(n, d, d_svm, d_x, d_y, d_alpha, d_g, lambda, i, j);
    cusmo_update_alpha<<<1,1>>>(d_y, d_alpha, i, j, lambda);
    cudaMemcpy(&alpha[i], &d_alpha[i], sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(&alpha[j], &d_alpha[j], sizeof(double), cudaMemcpyDeviceToHost);
  }
  svm.fit(x, y, alpha);
}

/* modified gradient projection method implementation */
template<typename SVMT>
void mgp(SVMT& svm, const vector<double>& x, const vector<double>& y, double epsilon) {
  
}

#endif
