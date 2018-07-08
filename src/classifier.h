
#ifndef _CLASSIFIER_H_
#define _CLASSIFIER_H_

#include <vector>
#include <algorithm>
#include <ctime>
#include <iostream>

#include "svm.h"

using namespace std;

enum class SVC_TYPE {
	OAA,
	OAO,
	TWOCLASS
};

template<KERNEL_TYPE KT, SVC_TYPE ST>
struct SVC {
};

template<KERNEL_TYPE KT>
class SVC<KT, SVC_TYPE::OAA> {
public:
	SVC(int classes, int d, const Kernel<KT>& kernel) : _classes(classes), _d(d), _kernel(kernel) {
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
private:
	vector<SVM<KT>> _classifiers;
	const int _classes;
	const Kernel<KT> _kernel;
	const int _d;
};

template <KERNEL_TYPE KT>
class SVC<KT, SVC_TYPE::OAO> {
public:
	SVC(int classes, int d, const Kernel<KT>& kernel) : _classes(classes), _d(d), _kernel(kernel) {
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
						auto x_begin = begin(x);
						auto x_end = begin(x);
						advance(x_begin, k*_d);
						advance(x_end, (k+1)*_d);
						x1.insert(end(x1), x_begin, x_end);
					}
				}
				cb_before(i, j);
				_classifiers.emplace_back(_d, _kernel);
				unsigned int iterations = smo(_classifiers.back(), x1, y1, 0.001, C);
				cb_after(_classifiers.back().getSupportVectorCount(), iterations);
			}
		}
	}

	int predict(const double* x) const {
		vector<int> scores(_classes, 0);
		int k = 0;
		for (int i = 0; i < _classes-1; i++) {
			for (int j = 0; j < _classes; j++) {
				scores[_classifiers[k].decision(x) > 0 ? i : j]++;
				k++;
			}
		}
		return distance(begin(scores), max_element(begin(scores), end(scores)));
	}

	int getD() const {
		return _d;
	}
private:
	vector<SVM<KT>> _classifiers;
	const int _classes;
	const Kernel<KT> _kernel;
	const int _d;
};

template <KERNEL_TYPE KT>
class SVC<KT, SVC_TYPE::TWOCLASS> {
public:
	SVC(int d, const Kernel<KT>& kernel) : _svm(d, kernel) {
	}

	template<typename F, typename G>
	void train(const vector<double>& x, const vector<int>& y, const double C, F cb_before, G cb_after) {
		unsigned int d = x.size() / y.size();
		vector<int> y1(y.size());
		transform(begin(y), end(y), begin(y1), [](int y_i) { return y_i == 0 ? 1 : -1; });
		cb_before(0, 1);
		unsigned int iterations = smo(_svm, x, y1, 0.001, C);
		cb_after(_svm.getSupportVectorCount(), iterations);
	}

	int predict(const double* x) const {
		return _svm.decision(x) > 0 ? 0 : 1;
	}

	int getD() const {
		return _svm.getD();
	}
private:
	SVM<KT> _svm;
};

template<KERNEL_TYPE KT, SVC_TYPE SVCT>
unsigned int testSVC(const SVC<KT, SVCT>& svc, const vector<double>& x, const vector<int>& y) {
	unsigned int hits = 0;
	for (unsigned int i = 0; i < y.size(); ++i) {
		int prediction = svc.predict(&x[i * x.size() / y.size()]);
		if (prediction == y[i])
			++hits;
	}
	return hits;
}

#endif
