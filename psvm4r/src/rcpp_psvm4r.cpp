#include <Rcpp.h>
#include <psvm/classifier.h>
#include <psvm/kernels.h>

using namespace Rcpp;

// [[Rcpp::export(name=".LinearKernel__new")]]
SEXP LinearKernel__new() {
	return Rcpp::XPtr<LinearKernel>(new LinearKernel(), true);
}
// [[Rcpp::export(name=".PolynomialKernel__new")]]
SEXP PolynomialKernel__new(double degree, double constant) {
	return Rcpp::XPtr<PolynomialKernel>(new PolynomialKernel(degree, constant), true);
}
// [[Rcpp::export(name=".RbfKernel__new")]]
SEXP RbfKernel__new(double gamma) {
	return Rcpp::XPtr<RbfKernel>(new RbfKernel(gamma), true);
}

// [[Rcpp::export(name=".Kernel__get")]]
double LinearKernel__get(SEXP xp, const NumericVector& a, const NumericVector& b) {
	Rcpp::XPtr<Kernel> ptr(xp);

	std::vector<double> a_copy = Rcpp::as<std::vector<double>>(a);
	std::vector<double> b_copy = Rcpp::as<std::vector<double>>(b);
	size_t min_size = std::min(a_copy.size(), b_copy.size());
	if (min_size == 0)
		return 0.0;
	return ptr->operator()(a_copy.data(), a_copy.data()+min_size, b_copy.data());
}

// [[Rcpp::export(name=".OneAgainstAllCSVC__new")]]
SEXP OneAgainstAllCSVC__new(int num_classes, int num_dimensions, SEXP xp) {
	Rcpp::XPtr<Kernel> kptr(xp);
	return Rcpp::XPtr<OneAgainstAllCSVC>(new OneAgainstAllCSVC(num_classes, num_dimensions, std::shared_ptr<const Kernel>(kptr->clone())), true);
}
// [[Rcpp::export(name=".OneAgainstOneCSVC__new")]]
SEXP OneAgainstOneCSVC__new(int num_classes, int num_dimensions, SEXP xp) {
	Rcpp::XPtr<Kernel> kptr(xp);
	return Rcpp::XPtr<OneAgainstOneCSVC>(new OneAgainstOneCSVC(num_classes, num_dimensions, std::shared_ptr<const Kernel>(kptr->clone())), true);
}

// [[Rcpp::export(name=".CSVC__train")]]
void CSVC__train(SEXP xp, const NumericVector& training_attributes, const IntegerVector& training_labels, double C) {
	Rcpp::XPtr<CSVC> ptr(xp);

	std::vector<double> attributes_copy = Rcpp::as<std::vector<double>>(training_attributes);
	std::vector<int> labels_copy = Rcpp::as<std::vector<int>>(training_labels);
	ptr->train(attributes_copy, labels_copy, C); 
}

// [[Rcpp::export(name=".CSVC__predict")]]
IntegerVector CSVC__predict(SEXP xp, const NumericVector& attributes) {
	Rcpp::XPtr<CSVC> ptr(xp);

	IntegerVector predictions(attributes.size() / ptr->num_dimensions, NA_INTEGER);
	std::vector<double> attributes_copy = Rcpp::as<std::vector<double>>(attributes);
	for (unsigned i = 0; i < predictions.size(); ++i) {
		predictions[i] = ptr->predict(&attributes_copy[i * ptr->num_dimensions]);
	}
	return predictions;
}

// [[Rcpp::export(name=".CSVC__num_sv")]]
IntegerVector SVC__num_sv(SEXP xp) {
	Rcpp::XPtr<CSVC> ptr(xp);
	IntegerVector result(ptr->classifiers.size());
	for (size_t i = 0; i < ptr->classifiers.size(); ++i)
		result[i] = ptr->classifiers[i].getSupportVectorCount();
	return result;
}

// [[Rcpp::export(name=".CSVC__num_classes")]]
int SVC__num_classes(SEXP xp) {
	Rcpp::XPtr<CSVC> ptr(xp);
	return ptr->num_classes;
}

// [[Rcpp::export(name=".CSVC__num_dimensions")]]
int SVC__num_dimensions(SEXP xp) {
	Rcpp::XPtr<CSVC> ptr(xp);
	return ptr->num_dimensions;
}

