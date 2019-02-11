#include <Rcpp.h>
#include <psvm/classifier.h>
#include <psvm/kernels.h>

using namespace Rcpp;

template <typename KT>
double K__get(SEXP xp, const NumericVector& a, const NumericVector& b) {
	Rcpp::XPtr<KT> ptr(xp);

	std::vector<double> a_copy = Rcpp::as<std::vector<double>>(a);
	std::vector<double> b_copy = Rcpp::as<std::vector<double>>(b);
	size_t min_size = std::min(a_copy.size(), b_copy.size());
	if (min_size == 0)
		return 0.0;
	return ptr->operator()(a_copy.data(), a_copy.data()+min_size, b_copy.data());
}

template <typename KT, template <class> typename SVCT>
SEXP SVC__new(int num_classes, int num_dimensions, SEXP kxp) {
	Rcpp::XPtr<KT> kptr(kxp);
	return Rcpp::XPtr<SVCT<KT>>(new SVCT<KT>(num_classes, num_dimensions, *kptr), true);
}

template <typename KT, template <class> typename SVCT>
void SVC__train(SEXP xp, const NumericVector& training_attributes, const IntegerVector& training_labels, double C) {
	Rcpp::XPtr<SVCT<KT>> ptr(xp);

	std::vector<double> attributes_copy = Rcpp::as<std::vector<double>>(training_attributes);
	std::vector<int> labels_copy = Rcpp::as<std::vector<int>>(training_labels);
	ptr->train(attributes_copy, labels_copy, C); 
}

template <typename KT, template <class> typename SVCT>
IntegerVector SVC__predict(SEXP xp, const NumericVector& attributes) {
	Rcpp::XPtr<SVCT<KT>> ptr(xp);

	IntegerVector predictions(attributes.size() / ptr->num_dimensions, NA_INTEGER);
	std::vector<double> attributes_copy = Rcpp::as<std::vector<double>>(attributes);
	for (unsigned i = 0; i < predictions.size(); ++i) {
		predictions[i] = ptr->predict(&attributes_copy[i * ptr->num_dimensions]);
	}
	return predictions;
}

template <typename KT, template <class> typename SVCT>
IntegerVector SVC__num_sv(SEXP xp) {
	Rcpp::XPtr<SVCT<KT>> ptr(xp);
	IntegerVector result(ptr->classifiers.size());
	for (size_t i = 0; i < ptr->classifiers.size(); ++i)
		result[i] = ptr->classifiers[i].getSupportVectorCount();
	return result;
}

template <typename KT, template <class> typename SVCT>
int SVC__num_classes(SEXP xp) {
	Rcpp::XPtr<SVCT<KT>> ptr(xp);
	return ptr->num_classes;
}

template <typename KT, template <class> typename SVCT>
int SVC__num_dimensions(SEXP xp) {
	Rcpp::XPtr<SVCT<KT>> ptr(xp);
	return ptr->num_dimensions;
}

// [[Rcpp::export(name=".LinearKernel__new")]]
SEXP LinearKernel__new() { return Rcpp::XPtr<LinearKernel>(new LinearKernel(), true); }
// [[Rcpp::export(name=".LinearKernel__get")]]
double LinearKernel__get(SEXP xp, const NumericVector& a, const NumericVector& b) { return K__get<LinearKernel>(xp, a, b); }

// [[Rcpp::export(name=".PolynomialKernel__new")]]
SEXP PolynomialKernel__new(double degree, double constant) { return Rcpp::XPtr<PolynomialKernel>(new PolynomialKernel(degree, constant), true); }
// [[Rcpp::export(name=".PolynomialKernel__get")]]
double PolynomialKernel__get(SEXP xp, const NumericVector& a, const NumericVector& b) { return K__get<PolynomialKernel>(xp, a, b); }

// [[Rcpp::export(name=".RbfKernel__new")]]
SEXP RbfKernel__new(double gamma) { return Rcpp::XPtr<RbfKernel>(new RbfKernel(gamma), true); }
// [[Rcpp::export(name=".RbfKernel__get")]]
double RbfKernel__get(SEXP xp, const NumericVector& a, const NumericVector& b) { return K__get<RbfKernel>(xp, a, b); }

// [[Rcpp::export(name=".OAALinearKernel__new")]]
SEXP OAALinearKernel__new(int num_classes, int num_dimensions, SEXP xp) { return SVC__new<LinearKernel, OneAgainstAllSVC>(num_classes, num_dimensions, xp); }
// [[Rcpp::export(name=".OAALinearKernel__train")]]
void OAALinearKernel__train(SEXP xp, const NumericVector& training_attributes, const IntegerVector& training_labels, double C) { SVC__train<LinearKernel, OneAgainstAllSVC>(xp, training_attributes, training_labels, C); }
// [[Rcpp::export(name=".OAALinearKernel__predict")]]
SEXP OAALinearKernel__predict(SEXP xp, NumericVector attributes) { return SVC__predict<LinearKernel, OneAgainstAllSVC>(xp, attributes); }
// [[Rcpp::export(name=".OAALinearKernel__num_sv")]]
SEXP OAALinearKernel__num_sv(SEXP xp) { return SVC__num_sv<LinearKernel, OneAgainstAllSVC>(xp); }
// [[Rcpp::export(name=".OAALinearKernel__num_classes")]]
int OAALinearKernel__num_classes(SEXP xp) { return SVC__num_classes<LinearKernel, OneAgainstAllSVC>(xp); }
// [[Rcpp::export(name=".OAALinearKernel__num_dimensions")]]
int OAALinearKernel__num_dimensions(SEXP xp) { return SVC__num_dimensions<LinearKernel, OneAgainstAllSVC>(xp); }

// [[Rcpp::export(name=".OAAPolynomialKernel__new")]]
SEXP OAAPolynomialKernel__new(int num_classes, int num_dimensions, SEXP xp) { return SVC__new<PolynomialKernel, OneAgainstAllSVC>(num_classes, num_dimensions, xp); }
// [[Rcpp::export(name=".OAAPolynomialKernel__get")]]
void OAAPolynomialKernel__train(SEXP xp, const NumericVector& training_attributes, const IntegerVector& training_labels, double C) { SVC__train<PolynomialKernel, OneAgainstAllSVC>(xp, training_attributes, training_labels, C); }
// [[Rcpp::export(name=".OAAPolynomialKernel__predict")]]
SEXP OAAPolynomialKernel__predict(SEXP xp, NumericVector attributes) { return SVC__predict<PolynomialKernel, OneAgainstAllSVC>(xp, attributes); }
// [[Rcpp::export(name=".OAAPolynomialKernel__num_sv")]]
SEXP OAAPolynomialKernel__num_sv(SEXP xp) { return SVC__num_sv<PolynomialKernel, OneAgainstAllSVC>(xp); }
// [[Rcpp::export(name=".OAAPolynomialKernel__num_classes")]]
int OAAPolynomialKernel__num_classes(SEXP xp) { return SVC__num_classes<PolynomialKernel, OneAgainstAllSVC>(xp); }
// [[Rcpp::export(name=".OAAPolynomialKernel__num_dimensions")]]
int OAAPolynomialKernel__num_dimensions(SEXP xp) { return SVC__num_dimensions<PolynomialKernel, OneAgainstAllSVC>(xp); }

// [[Rcpp::export(name=".OAARbfKernel__new")]]
SEXP OAARbfKernel__new(int num_classes, int num_dimensions, SEXP xp) { return SVC__new<RbfKernel, OneAgainstAllSVC>(num_classes, num_dimensions, xp); }
// [[Rcpp::export(name=".OAARbfKernel__train")]]
void OAARbfKernel__train(SEXP xp, const NumericVector& training_attributes, const IntegerVector& training_labels, double C) { SVC__train<RbfKernel, OneAgainstAllSVC>(xp, training_attributes, training_labels, C); }
// [[Rcpp::export(name=".OAARbfKernel__predict")]]
SEXP OAARbfKernel__predict(SEXP xp, NumericVector attributes) { return SVC__predict<RbfKernel, OneAgainstAllSVC>(xp, attributes); }
// [[Rcpp::export(name=".OAARbfKernel__num_sv")]]
SEXP OAARbfKernel__num_sv(SEXP xp) { return SVC__num_sv<RbfKernel, OneAgainstAllSVC>(xp); }
// [[Rcpp::export(name=".OAARbfKernel__num_classes")]]
int OAARbfKernel__num_classes(SEXP xp) { return SVC__num_classes<RbfKernel, OneAgainstAllSVC>(xp); }
// [[Rcpp::export(name=".OAARbfKernel__num_dimensions")]]
int OAARbfKernel__num_dimensions(SEXP xp) { return SVC__num_dimensions<RbfKernel, OneAgainstAllSVC>(xp); }

// [[Rcpp::export(name=".OAOLinearKernel__new")]]
SEXP OAOLinearKernel__new(int num_classes, int num_dimensions, SEXP xp) { return SVC__new<LinearKernel, OneAgainstOneSVC>(num_classes, num_dimensions, xp); }
// [[Rcpp::export(name=".OAOLinearKernel__train")]]
void OAOLinearKernel__train(SEXP xp, const NumericVector& training_attributes, const IntegerVector& training_labels, double C) { SVC__train<LinearKernel, OneAgainstOneSVC>(xp, training_attributes, training_labels, C); }
// [[Rcpp::export(name=".OAOLinearKernel__predict")]]
SEXP OAOLinearKernel__predict(SEXP xp, const NumericVector& attributes) { return SVC__predict<LinearKernel, OneAgainstOneSVC>(xp, attributes); }
// [[Rcpp::export(name=".OAOLinearKernel__num_sv")]]
SEXP OAOLinearKernel__num_sv(SEXP xp) { return SVC__num_sv<LinearKernel, OneAgainstOneSVC>(xp); }
// [[Rcpp::export(name=".OAOLinearKernel__num_classes")]]
int OAOLinearKernel__num_classes(SEXP xp) { return SVC__num_classes<LinearKernel, OneAgainstOneSVC>(xp); }
// [[Rcpp::export(name=".OAOLinearKernel__num_dimensions")]]
int OAOLinearKernel__num_dimensions(SEXP xp) { return SVC__num_dimensions<LinearKernel, OneAgainstOneSVC>(xp); }

// [[Rcpp::export(name=".OAOPolynomialKernel__new")]]
SEXP OAOPolynomialKernel__new(int num_classes, int num_dimensions, SEXP xp) { return SVC__new<PolynomialKernel, OneAgainstOneSVC>(num_classes, num_dimensions, xp); }
// [[Rcpp::export(name=".OAOPolynomialKernel__train")]]
void OAOPolynomialKernel__train(SEXP xp, const NumericVector& training_attributes, const IntegerVector& training_labels, double C) { SVC__train<PolynomialKernel, OneAgainstOneSVC>(xp, training_attributes, training_labels, C); }
// [[Rcpp::export(name=".OAOPolynomialKernel__predict")]]
SEXP OAOPolynomialKernel__predict(SEXP xp, const NumericVector& attributes) { return SVC__predict<PolynomialKernel, OneAgainstOneSVC>(xp, attributes); }
// [[Rcpp::export(name=".OAOPolynomialKernel__num_sv")]]
SEXP OAOPolynomialKernel__num_sv(SEXP xp) { return SVC__num_sv<PolynomialKernel, OneAgainstOneSVC>(xp); }
// [[Rcpp::export(name=".OAOPolynomialKernel__num_classes")]]
int OAOPolynomialKernel__num_classes(SEXP xp) { return SVC__num_classes<PolynomialKernel, OneAgainstOneSVC>(xp); }
// [[Rcpp::export(name=".OAOPolynomialKernel__num_dimensions")]]
int OAOPolynomialKernel__num_dimensions(SEXP xp) { return SVC__num_dimensions<PolynomialKernel, OneAgainstOneSVC>(xp); }

// [[Rcpp::export(name=".OAORbfKernel__new")]]
SEXP OAORbfKernel__new(int num_classes, int num_dimensions, SEXP xp) { return SVC__new<RbfKernel, OneAgainstOneSVC>(num_classes, num_dimensions, xp); }
// [[Rcpp::export(name=".OAORbfKernel__train")]]
void OAORbfKernel__train(SEXP xp, const NumericVector& training_attributes, const IntegerVector& training_labels, double C) { SVC__train<RbfKernel, OneAgainstOneSVC>(xp, training_attributes, training_labels, C); }
// [[Rcpp::export(name=".OAORbfKernel__predict")]]
SEXP OAORbfKernel__predict(SEXP xp, const NumericVector& attributes) { return SVC__predict<RbfKernel, OneAgainstOneSVC>(xp, attributes); }
// [[Rcpp::export(name=".OAORbfKernel__num_sv")]]
SEXP OAORbfKernel__num_sv(SEXP xp) { return SVC__num_sv<RbfKernel, OneAgainstOneSVC>(xp); }
// [[Rcpp::export(name=".OAORbfKernel__num_classes")]]
int OAORbfKernel__num_classes(SEXP xp) { return SVC__num_classes<RbfKernel, OneAgainstOneSVC>(xp); }
// [[Rcpp::export(name=".OAORbfKernel__num_dimensions")]]
int OAORbfKernel__num_dimensions(SEXP xp) { return SVC__num_dimensions<RbfKernel, OneAgainstOneSVC>(xp); }
