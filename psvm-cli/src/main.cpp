#include <chrono>
#include <vector>
#include <string>
#include <iostream>
#include <set>
#include <algorithm>
#include <fstream>

#include <boost/program_options.hpp>

#include <psvm/classifiers.h>
#include <psvm/model_io.h>

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
		("kernel-gamma",    po::value<double>()->default_value(.05), "gamma parameter for rbf kernel")
		("model-out,o",     po::value<std::string>(),         "output path to save model")
		("model-in,i",      po::value<std::string>(),         "input path to load model");

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

	const size_t num_attributes = x.size() / y.size();

	std::set<int> y_set(begin(y), end(y));
	const size_t num_classes = y_set.size();
	std::cout << x.size() << " training datapoints: " << num_attributes << " attributes x " << y.size() << " instances and " << num_classes << " classes" << std::endl;
	std::cout << test_x.size() << " testing datapoints" << std::endl;

	for (size_t i = 0; i < num_classes; i++) {
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
		double scale_min = 0.;
		double scale_max = 0.;
		if (options.count(NORMALIZATION_OPTION_1)) {
			scale_min = 0.0, scale_max = 1.0;
			std::cout << "Scaling attributes to [0;1] range" << std::endl;
		}
		else if (options.count(NORMALIZATION_OPTION_2)) {
			scale_min = -1.0, scale_max = 1.0;
			std::cout << "Scaling attributes to [-1; 1] range" << std::endl;
		}
		for (size_t i = 0; i < num_attributes; ++i) {
			double x_max = -std::numeric_limits<double>::infinity();
			double x_min = std::numeric_limits<double>::infinity();
			for (size_t j = 0; j < x.size(); j += num_attributes) {
				x_max = std::max(x_max, x[j+i]);
				x_min = std::min(x_min, x[j+i]);
			}

			for (auto dataset : datasets)
				for (size_t j = 0; j < dataset.get().size(); j += num_attributes)
					dataset.get()[j + i] = normalization_scale(dataset.get()[j + i], scale_min, scale_max, x_min, x_max);
		}
	}
	// normalize using z-score
	else if (options.count(NORMALIZATION_OPTION_Z)) {
		std::cout << "Normalizing attributes using z-score" << std::endl;
		for (size_t i = 0; i < num_attributes; ++i) {
			double x_mean = 0.0;
			double x_variance = 0.0;
			// calculate mean and stdev in two steps
			// there are algorithms for calculating stdev in one pass
			// but usually at the expense of numerical accuracy
			for (size_t j = 0; j < x.size(); j += num_attributes)
				x_mean += x[i+j] / static_cast<double>(y.size());
			for (size_t j = 0; j < x.size(); j += num_attributes)
				x_variance += (x[i+j] - x_mean) * (x[i+j] - x_mean) / static_cast<double>(y.size());

			for (auto dataset : datasets)
				for (size_t j = 0; j < dataset.get().size(); j += num_attributes)
					dataset.get()[j + i] = normalization_standard(dataset.get()[j + i], x_mean, std::sqrt(x_variance));
		}
	}

	// build kernel
	std::shared_ptr<const Kernel> kernel;
	if (options.count(KERNEL_OPTION_LINEAR)) {
		std::cout << "Using linear kernel" << std::endl;
		kernel = std::make_shared<const LinearKernel>();
	}
	else if (options.count(KERNEL_OPTION_POLYNOMIAL)) {
		double degree = options["kernel-degree"].as<double>();
		double constant = options["kernel-constant"].as<double>();
		std::cout << "Using Poly kernel with d=" << degree << ", c=" << constant << std::endl;
		kernel = std::make_shared<const PolynomialKernel>(degree, constant);
	}
	else if (options.count(KERNEL_OPTION_RBF)) {
		double gamma = options["kernel-gamma"].as<double>();
		std::cout << "Using RBF kernel with gamma=" << gamma << std::endl;
		kernel = std::make_shared<const RbfKernel>(gamma);
	}
	else {
		std::cerr << "Unknown kernel selection" << std::endl;
		return 1;
	}

	// build svc
	std::shared_ptr<CSVC> svc;
	if (options.count("model-in")) {
		std::string path = options["model-in"].as<std::string>();
		std::fstream f(path, std::ios_base::in);
		svc.reset(csvcFromStream(f));
		if (svc) {
			std::cout << "Loaded model from `" << path << "`\n";
		}
		else {
			std::cerr << "Failed to load model from `" << path << "`\n";
			return 1;
		}
	}
	else if (options.count(SVC_OPTION_1AA)) {
		std::cout << "One-against-all classification" << std::endl;
		svc = std::make_shared<OneAgainstAllCSVC>(num_classes, num_attributes, kernel);
		auto start_t = std::chrono::system_clock::now();
		svc->train(x, y, C,
			[&start_t](int i, size_t psize) {
				std::cout << "Training " << i << " vs. all (problem size: " << psize << ")" << std::endl;
				start_t = std::chrono::system_clock::now();
				return false;
			},
			[&start_t](const SVM& svm, unsigned int iters) {
				auto end_t = std::chrono::system_clock::now();
				std::chrono::duration<double> elapsed = end_t-start_t;
				std::cout << "#SVs: " << svm.getSupportVectorCount() << "(" << iters << " iterations in " << elapsed.count() << "s)" << std::endl;
				return false;
			}
		);
	}
	else if (options.count(SVC_OPTION_1A1)) {
		std::cout << "One-against-one classification" << std::endl;
		svc = std::make_shared<OneAgainstOneCSVC>(num_classes, num_attributes, kernel);
		auto start_t = std::chrono::system_clock::now();
		svc->train(x, y, C,
			[&start_t, num_classes](int i, size_t psize) {
				std::cout << "Training " << (i / num_classes) << " vs. " << (i % num_classes) << " (problem size: " << psize << ")" << std::endl;
				start_t = std::chrono::system_clock::now();
				return false;
			},
			[&start_t](const SVM& svm, unsigned int iters) {
				auto end_t = std::chrono::system_clock::now();
				std::chrono::duration<double> elapsed = end_t-start_t;
				std::cout << "#SVs: " << svm.getSupportVectorCount() << " (" << iters << " iterations in " << elapsed.count() << "s)" << std::endl;
				return false;
			}
		);
	}
	else {
		std::cerr << "Unknown svc selection" << std::endl;
		return 1;
	}

	if (options.count("model-out")) {
		std::string path = options["model-out"].as<std::string>();
		std::fstream f(path, std::ios_base::out);
		if (f) {
			toStream(svc.get(), f);
			std::cout << "Model saved to `" << path << "`\n";
		}
		else {
			std::cerr << "Could not save model to `" << path << "`\n";
		}
	}

	// testing
	// tests results are shown as a confusion matrix
	if (test_y.size() > 0) {
		std::vector<int> confusion_matrix (num_classes * num_classes, 0);
		for (size_t i = 0; i < test_y.size(); ++i) {
			int prediction = svc->predict(&test_x[i * num_attributes]);
			confusion_matrix[test_y[i] * num_classes + prediction]++;
		}
		// true positives + true negatives
		int tptn = 0;
		for (size_t i = 0; i < num_classes; ++i)
			tptn += confusion_matrix[i * num_classes+i];
		double accuracy = double(tptn) / double(test_y.size());
		std::cout << "Model accuracy: " << tptn << "/" << test_y.size() << " = " << accuracy << std::endl;
		std::cout << "Confusion matrix:" << std::endl;
		std::cout << "*\t";
		for (size_t j = 0; j < num_classes; j++) {
			std::cout << j << "\t";
		}
		std::cout << std::endl;
		for (size_t i = 0; i < num_classes; i++) {
			std::cout << i << "\t";
			for (size_t j = 0; j < num_classes; j++) {
				std::cout << confusion_matrix[i * num_classes + j] << "\t";
			}
			std::cout << std::endl;
		}
	}

	// predictions
	// predictions are shown line by line
	if (predict_x.size() > 0) {
		std::cout << "Predictions:" << std::endl;
		for (size_t i = 0; i < predict_x.size() / num_attributes; ++i) {
			std::cout << i << ". " << svc->predict(&predict_x[i * num_attributes]) << std::endl;
		}
	}
}

