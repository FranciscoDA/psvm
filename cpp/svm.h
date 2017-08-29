
#ifndef _SVM_H_
#define _SVM_H_

#ifdef __CUDACC__
  #define CUDA_CALLABLE_MEMBER __host__ __device__
#else
  #define CUDA_CALLABLE_MEMBER
#endif

#include <vector>
#include <cmath>

using namespace std;

class LinearKernel {
public:
  CUDA_CALLABLE_MEMBER double K(const double* x1, const double* x2, size_t d) const
  {
    double result = 0.0;
    for (size_t i = 0; i < d; ++i)
      result += x1[i] * x2[i];
    return result;
  }
};

class RbfKernel {
public:
  RbfKernel(double gamma) : _gamma(gamma)
  {}
  CUDA_CALLABLE_MEMBER double K(const double* x1, const double* x2, size_t d) const
  {
    double result = 0.0;
    for (size_t i = 0; i < d; ++i)
      result += (x1[i] - x2[i]) * (x1[i] - x2[i]);
    return exp(_gamma * result);
  }
private:
  double _gamma;
};

class PolynomialKernel {
public:
  PolynomialKernel(double d, double c, double a) : _d(d), _c(c), _a(a)
  {}
  CUDA_CALLABLE_MEMBER double K(const double* x1, const double* x2, size_t d) const
  {
    double result = 0.0;
    for (size_t i = 0; i < d; ++i)
      result += x1[i] * x2[i];
    return pow(result * _a + _c, _d);
  }
private:
  double _c;
  double _d;
  double _a;
};

template<typename K>
class SVM {
public:
  SVM(double C, size_t d, const K& k) : _C(C), _d(d), _kernel(k)
  {
  }
  double getC() const {
    return _C;
  }
  size_t getD() const {
    return _d;
  }
  void fit(const std::vector<double>& x, const std::vector<double>& y, const std::vector<double>& alpha) {
    _sv_x.clear();
    _sv_alpha_y.clear();
    for (int i = 0; i < y.size(); ++i) {
      if (alpha[i] > 0) {
        _sv_alpha_y.push_back(y[i] * alpha[i]);
        _sv_x.insert(end(_sv_x), begin(x) + i*_d, begin(x) + (i+1)*_d);
      }
    }
    // calculate b from a support vector that lies on a margin (ie: 0 < alpha_i < C)
    _b = 0;
    for (int i = 0; i < _sv_alpha_y.size(); ++i) {
      if (_sv_alpha_y[i] < _C) {
        // sum alpha_i y_i K(x_j,sv_i) + b = 1
        // iff: 1 - sum alpha_i y_i K(x_j, sv_i) = b
        _b = 1 - decision(&_sv_x[i * _d]);
        break;
      }
    }
  }
  double decision(const double* x) const {
    double sum = 0.0;
    for (int i = 0; i < _sv_alpha_y.size(); ++i) {
      sum += _sv_alpha_y[i] * kernel(&_sv_x[i*_d], x);
    }
    return sum+_b;
  }

  bool predict(const double* x) const {
    return decision(x,_d) > 0;
  }

  CUDA_CALLABLE_MEMBER double kernel(const double* x1, const double* x2) const {
    return _kernel.K(x1, x2, _d);
  }

  //double loss(const Example& x) const;

  CUDA_CALLABLE_MEMBER double A(const double y_i) const {
    // return 0 if S_i.y > 0 else -C
    return (_C * y_i - _C)/2.0;
  }
  CUDA_CALLABLE_MEMBER double B(const double y_i) const {
    // return C if S_i.y > 0 else 0
    return (_C * y_i + _C)/2.0;
  }

  bool dual_feasible(double alpha_i)
  {
    return 0.0 <= alpha_i and alpha_i <= _C;
  }
  bool dual_feasible(double alpha_i, double epsilon)
  {
    return 0.0-epsilon <= alpha_i and alpha_i <= _C+epsilon;
  }
  const std::vector<double>& getSVX() const {
    return _sv_x;
  }
  const std::vector<double>& getSVAlphaY() const {
    return _sv_alpha_y;
  }
private:
  /* TODO: Implement */
  //double hingeLoss(const Example& x) const;
  double _C;
  size_t _d;
  double _b;
  std::vector<double> _sv_alpha_y; // alpha_i * y_i of each sv
  std::vector<double> _sv_x;       // x_i of each sv
  const K _kernel;
};

#endif
