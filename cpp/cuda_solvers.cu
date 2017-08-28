
#include <iostream>
#include <limits>
#include <numeric>
#include <fstream>
#include <sstream>
#include <string>

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
  extern __shared__ H sdata[];

  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i >= n)
    return;

  sdata[tid] = mapping(i);
  __syncthreads();

  for (unsigned int s = 1; s < blockDim.x; s *= 2)
  {
    int index = s * 2 * tid;
    if(index < blockDim.x)
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

  double Kik = g_svm->kernel(&g_x[j*d], &g_x[k*d], d),
         Kjk = g_svm->kernel(&g_x[i*d], &g_x[k*d], d);
  g_g[k] += lambda * g_y[k] * (Kjk - Kik);
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
    auto result_i = *begin(gather_result);
    for (auto it = begin(gather_result); it != end(gather_result); ++it)
      result_i = reduce_i(result_i, *it);

    cu_mr<<<NUM_BLOCKS, BLOCK_SIZE>>>(n, d_result, map_j, reduce_j);
    cudaMemcpy(gather_result.data(), d_result, sizeof(Search_t) * gather_result.size(), cudaMemcpyDeviceToHost);
    auto result_j = *begin(gather_result);
    for (auto it = begin(gather_result); it != end(gather_result); ++it)
      result_j = reduce_j(result_j, *it);

    int i = result_i.index;
    double i_max = result_i.score;
    int j = result_j.index;
    double j_min = result_j.score;
    cout << "i: " << i << " imax: " << i_max << endl;
    cout << "j: " << j << " jmin: " << j_min << endl;
    if (i_max <= j_min)
      break;

    double Kii = svm.kernel(&x[i*d], &x[i*d], d),
           Kij = svm.kernel(&x[i*d], &x[j*d], d),
           Kjj = svm.kernel(&x[j*d], &x[j*d], d);

    double lambda = min(svm.B(y[i]) - y[i] * alpha[i], y[j] * alpha[j] - svm.A(y[j]));
    lambda = min(lambda, (i_max-j_min)/(Kii+Kjj-2*Kij));

    cusmo_update_gradient<<<NUM_BLOCKS, BLOCK_SIZE>>>(n, d, d_svm, d_x, d_y, d_alpha, d_g, lambda, i, j);

    alpha[i] += y[i] * lambda;
    alpha[j] -= y[j] * lambda;
    cudaMemcpy(&d_alpha[i], &alpha[i], sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(&d_alpha[j], &alpha[j], sizeof(double), cudaMemcpyHostToDevice);
  }
  svm.fit(x, y, alpha);
  for (int i = 0; i < n; i++)
    cout << "alpha_" << i << " = " << alpha[i] << endl;
}

int main(int argc, char** argv)
{
  int nCudaDevices;
  cudaGetDeviceCount(&nCudaDevices);
  cout << "Device count: " << nCudaDevices << endl;
  for (int i = 0; i < nCudaDevices; i++) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, i);
    cout << "Device #" << i << endl;
    cout << "\tName: " << prop.name << endl;
    cout << "\tMemory Clock (KHz): " << prop.memoryClockRate << endl;
    cout << "\tMemory Bus Width (bits): " << prop.memoryBusWidth << endl;
  }

  SVM<LinearKernel> svm(10.0, LinearKernel());
  vector<double> x;
  vector<double> y;

  fstream dataset("dataset.csv", ios_base::in);
  while (!dataset.eof()) {
    string line;
    dataset >> line;
    if (!line.length())
      continue;
    stringstream ss(line);
    double val;
    ss >> val;
    ss.ignore(1, ',');
    while (!ss.eof()) {
      x.push_back(val);
      ss >> val;
      ss.ignore(1, ',');
    }
    y.push_back(val);
  }
  if (x.size() % y.size() != 0) {
    cerr << "length of attributes is not divisible by length of classes" << endl;
    return 1;
  }
  for (double y_i : y) {
    if (y_i != -1.0 and y_i != 1.0) {
      cerr << "invalid class value: " << y_i << endl;
      return 2;
    }
  }
  smo(svm, x, y);
  cout << endl;
  //mgp(svm, x, y, 0.0001);
}
