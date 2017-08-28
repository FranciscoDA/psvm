
#ifndef _SVM_H_
#define _SVM_H_

#ifdef __CUDACC__
  #define CUDA_CALLABLE_MEMBER __host__ __device__
#else
  #define CUDA_CALLABLE_MEMBER
#endif

#include <vector>
#include <cmath>

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
  SVM(double C, const K& k) : _C(C), _kernel(k)
  {
  }
  double getC() const {
    return _C;
  }
  void fit(const std::vector<double>& x, const std::vector<double>& y, const std::vector<double>& alpha)
  {
    int n = y.size();
    int d = x.size() / y.size();
    _w.resize(d);

    for (int i = 0; i < d; ++i) {
      _w[i] = 0.0;
      for (int j = 0; j < n; ++j)
        _w[i] += x[j*d+i] * y[j] * alpha[j];
    }

    for (int i = 0; i < n; ++i) {
      if (0 < alpha[i] and alpha[i] < _C) {
        _b = y[i] - kernel(&x[i*d], _w.data(), d);
        break;
      }
    }
  }
  double decision(const double* x, size_t d) const {
    return kernel(x,_w)+_b;
  }

  bool predict(const double* x, size_t d) const {
    return decision(x,d) > 0;
  }

  CUDA_CALLABLE_MEMBER double kernel(const double* x1, const double* x2, size_t d) const {
    return _kernel.K(x1, x2, d);
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
private:
  /* TODO: Implement */
  //double hingeLoss(const Example& x) const;
  double _C;
  double _b;
  std::vector<double> _w;
  const K _kernel;
};

#endif
