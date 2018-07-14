
#ifndef _CLASSIFIER_H_
#define _CLASSIFIER_H_

#include <vector>
#include <algorithm>

#include "svm.h"

using namespace std;

template<typename KT>
class OneAgainstAllSVC {
public:
	using kernel_type = KT;

	OneAgainstAllSVC(int classes, int d, const KT& kernel) : _classes(classes), _d(d), _kernel(kernel) {
	}

	template <typename F, typename G>
	void train(const vector<double>& x, const vector<int>& y, const double C, F cb_before, G cb_after) {
		vector<int> y1 (y.size());
		for (int label = 0; label < _classes; ++label) {
			transform(begin(y), end(y), begin(y1), [label](const int& y_i){ return y_i==label ? 1 : -1; });
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
	vector<SVM<KT>> _classifiers;
	const int _d;
};

template <typename KT>
class OneAgainstOneSVC {
public:
	using kernel_type = KT;

	OneAgainstOneSVC(int classes, int d, const KT& kernel) : _classes(classes), _d(d), _kernel(kernel) {
	}

	template<typename F, typename G>
	void train(const vector<double>& x, const vector<int>& y, const double C, F cb_before, G cb_after) {
		for (int i = 0; i < _classes-1; i++) {
			for (int j = i+1; j < _classes; j++) {
				vector<double> x1;
				vector<int> y1;
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
		vector<int> scores(_classes, 0);
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
	vector<SVM<KT>> _classifiers;
	const int _d;
};

#endif
