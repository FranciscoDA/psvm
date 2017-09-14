
#ifndef _CLASSIFIER_H_
#define _CLASSIFIER_H_

#include <vector>
#include <algorithm>
#include <ctime>
#include <iostream>
using namespace std;

enum SVC_TYPE {
	SVC_1AA,
	SVC_1A1
};

template<typename SVMT>
double decision(const SVMT& svm, const double* x) {
	double sum = svm.getBias();
	for (int i = 0; i < svm.getSVAlphaY().size(); ++i) {
		sum += svm.getSVAlphaY()[i] * svm.kernel(&svm.getSVX()[i*svm.getD()], x);
	}
	return sum;
}
template<typename SVMT>
double test(const SVMT& svm, const vector<double>& x, const vector<double>& y) {
	unsigned int hits = 0;
	for (unsigned int i = 0; i < y.size(); i++) {
		double d = decision(svm, &x[i * x.size() / y.size()]);
		if ((d > 0 and y[i] == 1) or (d < 0 and y[i] == -1))
			hits++;
	}
	return double(hits) / double(y.size());
}

template<typename SVMT, SVC_TYPE ST>
struct SVC {
};

template<typename SVMT>
struct SVC<SVMT, SVC_1AA> {
public:
	template <typename F, typename G, typename ...KARG>
	static void train(
		vector<SVMT>& classifiers, const vector<double>& classes,
		const vector<double>& x, const vector<double>& y, const double C,
		F cb_before, G cb_after, KARG... kargs
	) {
		vector<double> y1 (y.size());
		for (double label : classes) {
			transform(begin(y), end(y), begin(y1), [label](double y_i){ return y_i==label ? 1.0 : -1.0; });
			cb_before(label);
			SVMT svm(x.size() / y.size(), typename SVMT::kernel_type(kargs...));
			unsigned int iterations = smo(svm, x, y1, 0.01, C);
			cb_after(svm, iterations);
			classifiers.push_back(svm);
		}
	}
	static int predict(const vector<SVMT>& classifiers, size_t num_k, const double* x) {
		int best = 0;
		double best_score = decision(classifiers[0], x);
		for (unsigned int i = 1; i < classifiers.size(); ++i) {
			double score = decision(classifiers[i], x);
			if (score > best_score) {
				best_score = score;
				best = i;
			}
		}
		return best;
	}
};

template <typename SVMT>
struct SVC<SVMT, SVC_1A1> {
public:
	template<typename F, typename G, typename ...KARG>
	static void train(
		vector<SVMT>& classifiers, const vector<double>& classes,
		const vector<double>& x, const vector<double>& y, const double C,
		F cb_before, G cb_after, KARG... kargs
	) {
		unsigned int d = x.size() / y.size();
		for (int i = 0; i < classes.size()-1; i++) {
			for (int j = i+1; j < classes.size(); j++) {
				vector<double> x1;
				vector<double> y1;
				for (int k = 0; k < y.size(); k++) {
					if (y[k] == classes[i] or y[k] == classes[j]) {
						y1.push_back(y[k] == classes[i] ? 1.0 : -1.0);
						for (int l = 0; l < d; l++)
						 	x1.push_back(x[k*d+l]);
					}
				}
				cb_before(classes[i], classes[j]);
				SVMT svm(d, typename SVMT::kernel_type(kargs...));
				unsigned int iterations = smo(svm, x1, y1, 0.01, C);
				cb_after(svm, iterations);
				classifiers.push_back(svm);
			}
		}
	}
	static int predict(const vector<SVMT>& classifiers, size_t num_k, const double* x) {
		vector<size_t> scores(num_k, 0);
		for (unsigned int i = 0; i < classifiers.size(); ++i) {
			int j = i;
			int k_plus = 0;
			while (j >= num_k-k_plus-1) {
				j -= num_k-k_plus-1;
				k_plus++;
			}
			int k_minus = k_plus+j+1;
			double decision_value = decision(classifiers[i], x);
			scores[decision_value > 0 ? k_plus : k_minus]++;
		}
		return distance(begin(scores), max_element(begin(scores), end(scores)));
	}
};

template<typename SVMT, typename SVCT>
unsigned int testSVC(
	const vector<SVMT>& classifiers, size_t num_k, const vector<double>& x, const vector<double>& y) {
	unsigned int hits = 0;
	for (unsigned int i = 0; i < y.size(); ++i) {
		double prediction = double(SVCT::predict(classifiers, num_k, &x[i * x.size() / y.size()]));
		if (prediction == y[i])
			++hits;
	}
	return hits;
}

#endif
