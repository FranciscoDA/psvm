#pragma once

#include <vector>
#include <functional>

#include "svm.h"
#include "smo.h"

class CSVC {
public:
	CSVC() = delete;
	virtual ~CSVC();

	virtual bool train(const std::vector<double>& x, const std::vector<int>& y, const double C, std::function<bool(int, size_t)> cb_before, std::function<bool(const SVM&, unsigned)> cb_after) = 0;
	virtual int predict(const double* x) const = 0;

	bool train(const std::vector<double>& x, const std::vector<int>& y, const double C);

	const int num_classes;
	const size_t num_dimensions;
	std::shared_ptr<const Kernel> kernel;
	std::vector<SVM> classifiers;
protected:
	CSVC(int classes, size_t dimensions, std::shared_ptr<const Kernel> kernel);

	// call the correct smo template according to the actual type of the kernel
	// this is necessary because cuda doesnt allow dynamic polymorphism
	unsigned train_smo(SVM& svm, const std::vector<double>& attributes, const std::vector<int>& labels, double epsilon, double C);
};

class OneAgainstAllCSVC : public CSVC {
public:
	OneAgainstAllCSVC(int classes, int dimensions, std::shared_ptr<const Kernel> kernel);

	bool train(const std::vector<double>& x, const std::vector<int>& y, const double C, std::function<bool(int, size_t)> cb_before, std::function<bool(const SVM&, unsigned)> cb_after) override;
	int predict(const double* x) const override;
};

class OneAgainstOneCSVC : public CSVC {
public:
	OneAgainstOneCSVC(int classes, int dimensions, std::shared_ptr<const Kernel> kernel);

	bool train(const std::vector<double>& x, const std::vector<int>& y, const double C, std::function<bool(int, size_t)> cb_before, std::function<bool(const SVM&, unsigned)> cb_after) override;
	int predict(const double* x) const override;
};
