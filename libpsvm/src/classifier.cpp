#include <algorithm>
#include "psvm/classifier.h"

CSVC::CSVC(int classes, size_t dimensions, std::shared_ptr<const Kernel> kernel) : num_classes(classes), num_dimensions(dimensions), kernel(kernel) {
}
CSVC::~CSVC() {
}

bool CSVC::train(const std::vector<double>& x, const std::vector<int>& y, const double C) {
	return train(x, y, C, [](int, size_t) { return false; }, [](const SVM&, unsigned) { return false; });
}

unsigned CSVC::train_smo(SVM& svm, const std::vector<double>& attributes, const std::vector<int>& labels, double epsilon, double C) {
	if (dynamic_cast<const LinearKernel*>(kernel.get())) {
		return smo<LinearKernel>(svm, attributes, labels, epsilon, C);
	}
	else if (dynamic_cast<const PolynomialKernel*>(kernel.get())) {
		return smo<PolynomialKernel>(svm, attributes, labels, epsilon, C);
	}
	else if (dynamic_cast<const RbfKernel*>(kernel.get())) {
		return smo<RbfKernel>(svm, attributes, labels, epsilon, C);
	}
	return -1;
}

OneAgainstAllCSVC::OneAgainstAllCSVC(int classes, int dimensions, std::shared_ptr<const Kernel> kernel) : CSVC(classes, dimensions, kernel) {
}

bool OneAgainstAllCSVC::train(const std::vector<double>& x, const std::vector<int>& y, const double C, std::function<bool(int, size_t)> cb_before, std::function<bool(const SVM&, unsigned)> cb_after) {
	std::vector<int> y1 (y.size());
	for (int label = 0; label < num_classes; ++label) {
		std::transform(begin(y), end(y), begin(y1), [label](const int& y_i){ return y_i==label ? 1 : -1; });
		if (cb_before(label, y1.size()))
			return false;

		classifiers.emplace_back(num_dimensions, kernel);
		unsigned int iterations = train_smo(classifiers.back(), x, y1, 0.01, C);

		if (cb_after(classifiers.back(), iterations))
			return false;
	}
	return true;
}

int OneAgainstAllCSVC::predict(const double* x) const {
	return distance(
		begin(classifiers),
		max_element(begin(classifiers), end(classifiers), [x](const SVM& svm1, const SVM& svm2) {
			return svm1.decision(x) < svm2.decision(x);
		})
	);
}

OneAgainstOneCSVC::OneAgainstOneCSVC(int classes, int dimensions, std::shared_ptr<const Kernel> kernel) : CSVC(classes, dimensions, kernel) {
}

bool OneAgainstOneCSVC::train(const std::vector<double>& x, const std::vector<int>& y, const double C, std::function<bool(int, size_t)> cb_before, std::function<bool(const SVM&, unsigned)> cb_after) {
	for (int i = 0; i < num_classes-1; i++) {
		for (int j = i+1; j < num_classes; j++) {
			std::vector<double> x1;
			std::vector<int> y1;
			for (size_t k = 0; k < y.size(); k++) {
				if (y[k] == i or y[k] == j) {
					y1.push_back(y[k] == i ? 1 : -1);
					x1.insert(end(x1), begin(x) + k*num_dimensions, begin(x) + (k+1)*num_dimensions);
				}
			}
			if (cb_before(i * num_classes + j, y1.size()))
				return false;
			classifiers.emplace_back(num_dimensions, kernel);
			unsigned int iterations = train_smo(classifiers.back(), x1, y1, 0.01, C);
			if (cb_after(classifiers.back(), iterations))
				return false;
		}
	}
	return true;
}

int OneAgainstOneCSVC::predict(const double* x) const {
	std::vector<int> scores(num_classes, 0);
	int k = 0;
	for (int i = 0; i < num_classes-1; i++) {
		for (int j = i+1; j < num_classes; j++) {
			scores[(classifiers[k].decision(x) > 0. ? i : j)]++;
			k++;
		}
	}
	return distance(begin(scores), max_element(begin(scores), end(scores)));
}

