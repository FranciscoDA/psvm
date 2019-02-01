#pragma once

#include <vector>
#include <algorithm>

#include "svm.h"

template<typename KT>
class OneAgainstAllSVC {
public:
	using kernel_type = KT;

	OneAgainstAllSVC(int classes, int d, const KT& kernel) : _classes(classes), _kernel(kernel), _d(d) {
	}

	template <typename F, typename G>
	void train(const std::vector<double>& x, const std::vector<int>& y, const double C, F cb_before, G cb_after) {
		std::vector<int> y1 (y.size());
		for (int label = 0; label < _classes; ++label) {
			std::transform(begin(y), end(y), begin(y1), [label](const int& y_i){ return y_i==label ? 1 : -1; });
			cb_before(label);
			_classifiers.emplace_back(_d, _kernel);
			unsigned int iterations = smo(_classifiers.back(), x, y1, 0.001, C);
			cb_after(_classifiers.back().getSupportVectorCount(), iterations);
		}
	}

	int predict(const double* x) const {
		return distance(
			begin(_classifiers),
			max_element(begin(_classifiers), end(_classifiers), [x](const SVM<KT>& svm1, const SVM<KT>& svm2) {
				return svm1.decision(x) < svm2.decision(x);
			})
		);
	}

	int getD() const {
		return _d;
	}
	const int _classes;
	const KT _kernel;
private:
	std::vector<SVM<KT>> _classifiers;
	const int _d;
};

template <typename KT>
class OneAgainstOneSVC {
public:
	using kernel_type = KT;

	OneAgainstOneSVC(int classes, int d, const KT& kernel) : _classes(classes), _kernel(kernel), _d(d) {
	}

	template<typename F, typename G>
	void train(const std::vector<double>& x, const std::vector<int>& y, const double C, F cb_before, G cb_after) {
		for (int i = 0; i < _classes-1; i++) {
			for (int j = i+1; j < _classes; j++) {
				std::vector<double> x1;
				std::vector<int> y1;
				for (int k = 0; k < y.size(); k++) {
					if (y[k] == i or y[k] == j) {
						y1.push_back(y[k] == i ? 1 : -1);
						x1.insert(end(x1), begin(x) + k*_d, begin(x) + (k+1)*_d);
					}
				}
				cb_before(i, j, y1.size());
				_classifiers.emplace_back(_d, _kernel);
				unsigned int iterations = smo(_classifiers.back(), x1, y1, 0.001, C);
				cb_after(_classifiers.back().getSupportVectorCount(), _classifiers.back().getBias(), iterations);
			}
		}
	}

	int predict(const double* x) const {
		std::vector<int> scores(_classes, 0);
		int k = 0;
		for (int i = 0; i < _classes-1; i++) {
			for (int j = i+1; j < _classes; j++) {
				scores[(_classifiers[k].decision(x) > 0. ? i : j)]++;
				k++;
			}
		}
		return distance(begin(scores), max_element(begin(scores), end(scores)));
	}

	int getD() const {
		return _d;
	}
	const int _classes;
	const KT _kernel;
private:
	std::vector<SVM<KT>> _classifiers;
	const int _d;
};

#ifndef SUPPRESS_CLASSIFIER_EXTERN_TEMPLATES
extern template class OneAgainstAllSVC<LinearKernel>;
extern template class OneAgainstAllSVC<RbfKernel>;
extern template class OneAgainstAllSVC<PolynomialKernel>;

extern template class OneAgainstOneSVC<LinearKernel>;
extern template class OneAgainstOneSVC<RbfKernel>;
extern template class OneAgainstOneSVC<PolynomialKernel>;
#endif
