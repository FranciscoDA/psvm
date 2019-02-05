#include <chrono>
#include <vector>
#include <string>
#include <iostream>
#include <set>
#include <algorithm>

#include <boost/program_options.hpp>

#include <psvm/classifier.h>
#include <psvm/svm.h>

#include "io_formats.h"
#include "utils.h"

namespace po = boost::program_options;

static const char* KERNEL_OPTION_LINEAR     = "linear";
static const char* KERNEL_OPTION_POLYNOMIAL = "polynomial";
static const char* KERNEL_OPTION_RBF        = "rbf";
static const char* NORMALIZATION_OPTION_1   = "n1";
static const char* NORMALIZATION_OPTION_2   = "n2";
static const char* NORMALIZATION_OPTION_Z   = "nz";
static const char* SVC_OPTION_1AA           = "1AA";
static const char* SVC_OPTION_1A1           = "1A1";

// perform tests and predictions
// tests results are shown as a confusion matrix
// prediction results are shown line by line
template<typename SVCT>
void do_test_predict(
	const std::vector<double>& tx, const std::vector<int>& ty,
	const std::vector<double>& px, const SVCT& svc
) {
	if (ty.size() > 0) {
		std::vector<int> confusion_matrix (svc._classes * svc._classes);
		for (int i = 0; i < ty.size(); ++i) {
			int prediction = svc.predict(&tx[i * svc.getDimensions()]);
			confusion_matrix[ty[i] * svc._classes + prediction]++;
		}
		int tptn = 0;
		for (int i = 0; i < svc._classes; ++i)
			tptn += confusion_matrix[i*svc._classes+i];
		double accuracy = double(tptn)/double(ty.size());
		std::cout << "Model accuracy: " << tptn << "/" << ty.size() << " = " << accuracy << std::endl;
		std::cout << "Confusion matrix:" << std::endl;
		std::cout << "*\t";
		for (int j = 0; j < svc._classes; j++) {
			std::cout << j << "\t";
		}
		std::cout << std::endl;
		for (int i = 0; i < svc._classes; i++) {
			std::cout << i << "\t";
			for (int j = 0; j < svc._classes; j++) {
				std::cout << confusion_matrix[i*svc._classes+j] << "\t";
			}
			std::cout << std::endl;
		}
	}
	if (px.size() > 0) {
		std::cout << "Predictions:" << std::endl;
		for (int i = 0; i < px.size()/svc.getDimensions(); ++i) {
			std::cout << i << ". " << svc.predict(&px[i*svc.getDimensions()]) << std::endl;
		}
	}
}

// build and train classifier from the command-line options and call do_test_predict
template<typename KT>
void do_build_svc(
	const std::vector<double>& x, const std::vector<int>& y,
	const std::vector<double>& tx, const std::vector<int>& ty,
	const std::vector<double>& px, const double C, KT&& kernel, int num_attributes, int num_classes,
	const po::variables_map& options
) {
	if (options.count(SVC_OPTION_1AA)) {
		std::cout << "One-against-all classification" << std::endl;
		OneAgainstAllSVC<KT> svc (num_classes, num_attributes, kernel);
		auto start_t = std::chrono::system_clock::now();
		svc.train(x, y, C,
			[&start_t](int i) {
				std::cout << "Training " << i << " vs. all" << std::endl;
				start_t = std::chrono::system_clock::now();
			},
			[start_t](unsigned int nsvs, unsigned int iters) {
				auto end_t = std::chrono::system_clock::now();
				std::chrono::duration<double> elapsed = end_t-start_t;
				std::cout << "#SVs: " << nsvs << "(" << iters << " iterations in " << elapsed.count() << "s)" << std::endl;
			}
		);
		do_test_predict(tx, ty, px, svc);
	}
	else if (options.count(SVC_OPTION_1A1)) {
		std::cout << "One-against-one classification" << std::endl;
		OneAgainstOneSVC<KT> svc (num_classes, num_attributes, kernel);
		auto start_t = std::chrono::system_clock::now();
		svc.train(x, y, C,
			[&start_t](int i, int j, size_t psize) {
				std::cout << "Training " << i << " vs. " << j << " (problem size: " << psize << ")" << std::endl;
				start_t = std::chrono::system_clock::now();
			},
			[start_t](unsigned int nsvs, unsigned int iters) {
				auto end_t = std::chrono::system_clock::now();
				std::chrono::duration<double> elapsed = end_t-start_t;
				std::cout << "#SVs: " << nsvs << " (" << iters << " iterations in " << elapsed.count() << "s)" << std::endl;
			}
		);
		do_test_predict(tx, ty, px, svc);
	}
	else {
		std::cerr << "Unknown svc selection" << std::endl;
	}
}

// build the SVM kernel using the command-line options and call do_build_svc
void do_build_kernel(
	const std::vector<double>& x, const std::vector<int>& y,
	const std::vector<double>& tx, const std::vector<int>& ty,
	const std::vector<double>& px, const double C, int num_attributes, int num_classes,
	const po::variables_map& options
) {
	if (options.count(KERNEL_OPTION_LINEAR)) {
		std::cout << "Using linear kernel" << std::endl;
		do_build_svc(x, y, tx, ty, px, C, LinearKernel(), num_attributes, num_classes, options);
	}
	else if (options.count(KERNEL_OPTION_RBF)) {
		double gamma = options["kernel-gamma"].as<double>();
		std::cout << "Using RBF kernel with gamma=" << gamma << std::endl;
		do_build_svc(x, y, tx, ty, px, C, RbfKernel(gamma), num_attributes, num_classes, options);
	}
	else if (options.count(KERNEL_OPTION_POLYNOMIAL)) {
		double d = options["kernel-degree"].as<double>();
		double c = options["kernel-constant"].as<double>();
		std::cout << "Using Poly kernel with d=" << d << ", c=" << c << std::endl;
		do_build_svc(x, y, tx, ty, px, C, PolynomialKernel(d,C), num_attributes, num_classes, options);
	}
	else {
		std::cerr << "Unknown kernel selection" << std::endl;
	}
}

template<typename T>
void do_read_data(std::vector<T>& out, const std::string& option_path_key, const std::string& option_fmt_key, const po::variables_map& options) {
	// if option is set, use the format specified by the user. autodetect otherwise
	if (options.count(option_fmt_key))
		read_dataset(out, options[option_path_key].as<std::string>(), format_name_to_io_format(options[option_fmt_key].as<std::string>()));
	else
		read_dataset(out, options[option_path_key].as<std::string>());
}


int main(int argc, char** argv) {
	#ifdef __CUDACC__
	int nCudaDevices;
	cudaGetDeviceCount(&nCudaDevices);
	std::cout << "Device count: " << nCudaDevices << std::endl;
	for (int i = 0; i < nCudaDevices; i++) {
		cudaDeviceProp prop;
		cudaGetDeviceProperties(&prop, i);
		std::cout << "Device #" << i << "\n"
		          << "\tName: " << prop.name << "\n"
		          << "\tMemory Clock (KHz): " << prop.memoryClockRate << "\n"
		          << "\tMemory Bus Width (bits): " << prop.memoryBusWidth << std::endl;
	}
	#endif

	po::options_description desc("Allowed options:");
	desc.add_options()
		("help,h",                                            "show this help")
		("train-attributes",   po::value<std::string>()->required(), "path to training dataset with attributes (required)")
		("train-labels",       po::value<std::string>()->required(), "path to training dataset with labels (required)")
		("train-format",       po::value<std::string>(),      ("override dataset format detection (" + join_str("|", IO_FORMAT_NAMES) + ")").c_str())
		("test-attributes",    po::value<std::string>(),      "path to testing dataset with attributes (optional)")
		("test-labels",        po::value<std::string>(),      "path to testing dataset with labels (optional")
		("test-format",        po::value<std::string>(),      ("override dataset format detection (" + join_str("|", IO_FORMAT_NAMES) + ")").c_str())
		("predict-attributes", po::value<std::string>(),      "path to dataset with attributes to predict (optional)")
		("predict-format",     po::value<std::string>(),      ("override dataset format detection (" + join_str("|", IO_FORMAT_NAMES) + ")").c_str())
		(NORMALIZATION_OPTION_1,                              "normalize attributes to range [0; 1]")
		(NORMALIZATION_OPTION_2,                              "normalize attributes to range [-1; 1]")
		(NORMALIZATION_OPTION_Z,                              "normalize attributes with z-score")
		(KERNEL_OPTION_LINEAR,                                "use linear kernel: K(x, x') = x . x'")
		(KERNEL_OPTION_POLYNOMIAL,                            "use polynomial kernel: K(x, x') = (x . x' + constant)^degree")
		(KERNEL_OPTION_RBF,                                   "use gaussian kernel: K(x, x') = exp(-gamma * (x - x')^2)")
		(SVC_OPTION_1AA,                                      "train one-against-all multiclass classifier")
		(SVC_OPTION_1A1,                                      "train one-against-one multiclass classifier")
		("cost,C",          po::value<double>()->default_value(1.0), "set cost parameter")
		("kernel-degree",   po::value<double>()->default_value(2.0), "degree for polynomial kernel")
		("kernel-constant", po::value<double>()->default_value(0.0), "constant for polynomial kernel")
		("kernel-gamma",    po::value<double>()->default_value(.05), "gamma parameter for rbf kernel");

	po::variables_map options;
	po::store(po::parse_command_line(argc, argv, desc), options);

	if (options.count("help")) {
		std::cout << "Train, test and predict using SVMs" << "\n"
		          << desc << std::endl;
		return 0;
	}
	try {
		po::notify(options);
	}
	catch (const po::required_option& e) {
		std::cerr << e.what() << std::endl;
		return 1;
	}
	
	double C = options["cost"].as<double>();

	std::cout << "Using cost factor C=" << C << std::endl;

	std::vector<double> x;
	std::vector<int> y;

	std::vector<double> test_x;
	std::vector<int> test_y;

	std::vector<double> predict_x;

	std::cout << "Reading datasets..." << std::endl;
	try {
		if (options.count("train-attributes") and options.count("train-labels")) {
			do_read_data(x, "train-attributes", "train-format", options);
			do_read_data(y, "train-labels",     "train-format", options);
			if (y.size() == 0) {
				std::cerr << "DATASET ERROR: no labels found in " << options["train-labels"].as<std::string>() << std::endl;
				return 1;
			}
			else if (x.size() == 0) {
				std::cerr << "DATASET ERROR: no attributes found in " << options["train-attributes"].as<std::string>() << std::endl;
				return 1;
			}
			else if (x.size() % y.size() != 0) {
				std::cerr << "DATASET ERROR: number of training attributes not divisible by number of labels" << std::endl;
				return 1;
			}
		}
		if (options.count("test-attributes") and options.count("test-labels")) {
			do_read_data(test_x, "test-attributes", "test-format", options);
			do_read_data(test_y, "test-labels",     "test-format", options);
			if (test_y.size() == 0) {
				std::cerr << "DATASET ERROR: no labels found in " << options["train-labels"].as<std::string>() << std::endl;
				return 1;
			}
			else if (test_x.size() == 0) {
				std::cerr << "DATASET ERROR: no attributes found in " << options["train-attributes"].as<std::string>() << std::endl;
				return 1;
			}
			else if (test_x.size() % test_y.size() != 0) {
				std::cerr << "DATASET ERROR: number of testing attributes not divisible by number of labels" << std::endl;
				return 1;
			}
			else if (test_x.size() / test_y.size() != x.size() / y.size()) {
				std::cerr << "DATASET ERROR: number of testing testing attributes doesn't match number of training attributes" << std::endl;
				return 1;
			}
		}
		if (options.count("predict-attributes")) {
			do_read_data(predict_x, "predict-attributes", "predict-format", options);
			if (predict_x.size() == 0) {
				std::cerr << "DATASET ERROR: no attributes found in " << options["predict-attributes"].as<std::string>() << std::endl;
				return 1;
			}
			else if (predict_x.size() % (x.size() / y.size()) != 0) {
				std::cerr << "DATASET ERROR: number of predicting attributes doesn't match number of training attributes" << std::endl;
				return 1;
			}
		}
	}
	catch (DatasetError e) {
		std::cerr << "DATASET ERROR: ";
		switch (e.code) {
			case DatasetError::INCONSISTENT_ROWS:
				std::cerr << "number of columns in row doesn't match with the rest of the dataset";
				break;
			case DatasetError::HEADER_MISMATCH:
				std::cerr << "header mismatch";
				break;
			case DatasetError::INVALID_TYPE:
				std::cerr << "invalid data type";
				break;
			case DatasetError::UNKNOWN_FORMAT:
				std::cerr << "unknown dataset format";
				break;
			default:
				std::cerr << "error in dataset";
				break;
		}
		if (e.filename != "") {
			std::cerr << " in " << e.filename;
			if (e.position != 0)
				std::cerr << ":" << e.position;
		}
		std::cerr << std::endl;
		return 1;
	}

	const int num_attributes = x.size() / y.size();

	std::set<int> y_set(begin(y), end(y));
	const int num_classes = y_set.size();
	std::cout << x.size() << " training datapoints divided between " << y.size() << " instances and " << num_classes << " classes" << std::endl;
	std::cout << test_x.size() << " testing datapoints" << std::endl;

	for (int i = 0; i < num_classes; i++) {
		if (y_set.find(i) == end(y_set)) {
			std::cerr << "ERROR: classes are not contiguous" << std::endl;
			return 1;
		}
	}
	for (auto i : test_y) {
		if (y_set.find(i) == end(y_set)) {
			std::cerr << "ERROR: testing dataset contains classes not in training" << std::endl;
			return 1;
		}
	}

	std::vector<std::reference_wrapper<std::vector<double>>> datasets {x, test_x, predict_x};

	// normalize using feature scaling
	if (options.count(NORMALIZATION_OPTION_1) or options.count(NORMALIZATION_OPTION_2)) {
		double scale_min;
		double scale_max;
		if (options.count(NORMALIZATION_OPTION_1)) {
			scale_min = 0.0, scale_max = 1.0;
			std::cout << "Scaling attributes to [0;1] range" << std::endl;
		}
		else if (options.count(NORMALIZATION_OPTION_2)) {
			scale_min = -1.0, scale_max = 1.0;
			std::cout << "Scaling attributes to [-1; 1] range" << std::endl;
		}
		for (int i = 0; i < num_attributes; ++i) {
			double x_max = -std::numeric_limits<double>::infinity();
			double x_min = std::numeric_limits<double>::infinity();
			for (int j = 0; j < x.size(); j += num_attributes) {
				x_max = std::max(x_max, x[j+i]);
				x_min = std::min(x_min, x[j+i]);
			}

			for (auto dataset : datasets)
				for (int j = 0; j < dataset.get().size(); j += num_attributes)
					dataset.get()[j + i] = normalization_scale(dataset.get()[j + i], scale_min, scale_max, x_min, x_max);
		}
	}
	// normalize using z-score
	else if (options.count(NORMALIZATION_OPTION_Z)) {
		std::cout << "Normalizing attributes using z-score" << std::endl;
		double x_mean;
		double x_variance;
		for (int i = 0; i < num_attributes; ++i) {
			// calculate mean and stdev in two steps
			// there are algorithms for calculating stdev in one pass
			// but usually at the expense of numerical accuracy
			for (int j = 0; j < x.size(); j += num_attributes)
				x_mean += x[i+j] / static_cast<double>(y.size());
			for (int j = 0; j < x.size(); j += num_attributes)
				x_variance += (x[i+j] - x_mean) * (x[i+j] - x_mean) / static_cast<double>(y.size());

			for (auto dataset : datasets)
				for (int j = 0; j < dataset.get().size(); j += num_attributes)
					dataset.get()[j + i] = normalization_standard(dataset.get()[j + i], x_mean, std::sqrt(x_variance));
		}
	}

	do_build_kernel(x, y, test_x, test_y, predict_x, C, num_attributes, num_classes, options);
}

