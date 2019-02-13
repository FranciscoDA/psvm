#include "psvm/model_io.h"

#include <iterator>

Kernel* kernelFromStream(std::istream& stream) {
	std::string kernel_type;
	stream >> kernel_type;
	if (kernel_type == "LinearKernel") {
		return new LinearKernel();
	}
	else if (kernel_type == "PolynomialKernel") {
		double degree;
		double constant;
		stream >> degree;
		stream >> constant;
		return new PolynomialKernel(degree, constant);
	}
	else if (kernel_type == "RbfKernel") {
		double gamma;
		stream >> gamma;
		return new RbfKernel(gamma);
	}
	return nullptr;
}

CSVC* csvcFromStream(std::istream& stream) {
	std::string csvc_type;
	size_t num_classes;
	size_t num_dimensions;

	stream >> csvc_type;
	stream >> num_classes;
	stream >> num_dimensions;
	std::shared_ptr<Kernel> kernel {kernelFromStream(stream)};

	if (!kernel)
		return nullptr;

	CSVC* csvc = nullptr;
	size_t num_classifiers = 0;
	if (csvc_type == "OneAgainstAllCSVC") {
		csvc = new OneAgainstAllCSVC(num_classes, num_dimensions, kernel);
		num_classifiers = num_classes;
	}
	else if (csvc_type == "OneAgainstOneCSVC") {
		csvc = new OneAgainstOneCSVC(num_classes, num_dimensions, kernel);
		num_classifiers = num_classes * (num_classes-1) / 2;
	}
	if (!csvc)
		return nullptr;

	for (size_t i = 0; i < num_classifiers; ++i) {
		csvc->classifiers.emplace_back(num_dimensions, kernel);
		auto& svm = csvc->classifiers.back();
		stream >> svm.bias;
		size_t num_sv;
		stream >> num_sv;

		svm.sv_x.resize(num_sv * num_dimensions);
		svm.sv_alpha_y.resize(num_sv);

		for (size_t j = 0; j < svm.sv_x.size(); ++j)
			stream >> svm.sv_x[j];

		for (size_t j = 0; j < svm.sv_alpha_y.size(); ++j)
			stream >> svm.sv_alpha_y[j];
	}

	return csvc;
}

void toStream(const Kernel* kernel, std::ostream& stream) {
	if (dynamic_cast<const LinearKernel*>(kernel)) {
		stream << "LinearKernel\n";
	}
	else if (auto p = dynamic_cast<const PolynomialKernel*>(kernel)) {
		stream << "PolynomialKernel " << p->degree << " " << p->constant << "\n";
	}
	else if (auto p = dynamic_cast<const RbfKernel*>(kernel)) {
		stream << "RbfKernel " << p->gamma << "\n";
	}
}

void toStream(const CSVC* csvc, std::ostream& stream) {
	if (dynamic_cast<const OneAgainstAllCSVC*>(csvc)) {
		stream << "OneAgainstAllCSVC ";
	}
	else if (dynamic_cast<const OneAgainstOneCSVC*>(csvc)) {
		stream << "OneAgainstOneCSVC ";
	}
	stream << csvc->num_classes << " " << csvc->num_dimensions << "\n";
	toStream(csvc->kernel.get(), stream);
	for (const auto& svm : csvc->classifiers) {
		stream << svm.bias << " " << svm.sv_alpha_y.size() << "\n";
		if (!svm.sv_x.empty()){
			std::copy(svm.sv_x.cbegin(), --svm.sv_x.cend(), std::ostream_iterator<double>(stream, " "));
			stream << svm.sv_x.back();
		}
		stream << "\n";
		if (!svm.sv_alpha_y.empty()) {
			std::copy(svm.sv_alpha_y.cbegin(), --svm.sv_alpha_y.cend(), std::ostream_iterator<double>(stream, " "));
			stream << svm.sv_alpha_y.back();
		}
		stream << "\n";
	}
}
