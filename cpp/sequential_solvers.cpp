
#ifndef _SEQUENTIAL_SOLVERS_H_
#define _SEQUENTIAL_SOLVERS_H_

#include <limits>
#include <numeric>

#include "svm.h"

using namespace std;

/* sequential minimal optimization method implementation */
template<typename SVMT>
void smo(SVMT& svm, const vector<double>& x, const vector<double>& y) {
  size_t n = y.size();
  size_t d = svm.getD();
  vector<double> alpha(n, 0.0);
  vector<double> g(n, 1.0);

  while(true) {
    int i = 0, j = 0;
    double i_max = -numeric_limits<double>::infinity(),
           j_min = numeric_limits<double>::infinity();

    for (int k = 0; k < n; k++) {
      if (y[k] * alpha[k] < svm.B(y[k]) and i_max < y[k] * g[k]) {
        i = k;
        i_max = y[k] * g[k];
      }
      if (svm.A(y[k]) < y[k] * alpha[k] and y[k] * g[k] < j_min) {
        j = k;
        j_min = y[k] * g[k];
      }
    }
    if (i_max <= j_min)
      break;

    double Kii = svm.kernel(&x[i*d], &x[i*d]),
           Kij = svm.kernel(&x[i*d], &x[j*d]),
           Kjj = svm.kernel(&x[j*d], &x[j*d]);

    double lambda = min(svm.B(y[i]) - y[i] * alpha[i], y[j] * alpha[j] - svm.A(y[j]));
    lambda = min(lambda, (i_max-j_min)/(Kii+Kjj-2*Kij));

    for(int k = 0; k < n; k++) {
      double Kik = svm.kernel(&x[i*d], &x[k*d]),
             Kjk = svm.kernel(&x[j*d], &x[k*d]);
      g[k] += lambda * y[k] * (Kjk - Kik);
    }
    alpha[i] += y[i] * lambda;
    alpha[j] -= y[j] * lambda;
  }
  svm.fit(x, y, alpha);
}

/* modified gradient projection method implementation */
template<typename SVMT>
void mgp(SVMT& svm, const vector<double>& x, const vector<double>& y, double epsilon) {
  size_t n = y.size();
  size_t d = svm.getD();
  vector<double> alpha(n, 0.0);
  vector<double> g(n, 0.0);
  vector<double> u(n, 0.0);

  vector<int> violating_samples(n);
  for (int i = 0; i < violating_samples.size(); ++i)
    violating_samples[i] = i;

  while(violating_samples.size() > 0) {
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
          if (u[*it] > 0 and alpha[*it] >= svm.getC() or u[*it] < 0 and alpha[*it] <= 0)
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
        lambda_max = min(lambda_max, u[i] != 0 ? max(0.0, svm.getC() / u[i]) - alpha[i] / u[i] : lambda_max);
      double lambda_star = max(0.0, min(lambda_max, gu/uhu));

      for (int i = 0; i < n; i++)
        alpha[i] += lambda_star * u[i];
    }
    violating_samples.clear();
    for (int i = 0; i < n; i++)
      if (!svm.dual_feasible(alpha[i], epsilon))
        violating_samples.push_back(i);
  }
  svm.fit(x, y, alpha);
}

#endif
