
#include <chrono>
#include <vector>
#include <string>
#include <iostream>
#include <set>
#include <algorithm>

#include "optparse/optparse.h"

#include "classifier.h"
#include "io_formats.h"
#include "iterutils/strided_range.h"
#include "iterutils/ref_range.h"
#include "utils.h"

#include "svm.h"

#ifdef __CUDACC__
#include "cuda_solvers.cu"
#else
#include "sequential_solvers.h"
#endif

#include <CL/sycl.hpp>

using namespace std;

template<typename SVCT>
void do_test_predict(
	const vector<double>& tx, const vector<int>& ty,
	const vector<double>& px, const SVCT& svc
) {
	if (ty.size() > 0) {
		vector<int> confusion_matrix (svc._classes * svc._classes);
		for (int i = 0; i < ty.size(); ++i) {
			int prediction = svc.predict(&tx[i * svc.getD()]);
			confusion_matrix[ty[i] * svc._classes + prediction]++;
		}
		int tptn = 0;
		for (int i = 0; i < svc._classes; ++i)
			tptn += confusion_matrix[i*svc._classes+i];
		double accuracy = double(tptn)/double(ty.size());
		cout << "Model accuracy: " << tptn << "/" << ty.size() << " = " << accuracy << endl;
		cout << "Confusion matrix:" << endl;
		cout << "*\t";
		for (int j = 0; j < svc._classes; j++) {
			cout << j << "\t";
		}
		cout << endl;
		for (int i = 0; i < svc._classes; i++) {
			cout << i << "\t";
			for (int j = 0; j < svc._classes; j++) {
				cout << confusion_matrix[i*svc._classes+j] << "\t";
			}
			cout << endl;
		}
	}
	if (px.size() > 0) {
		cout << "Predictions:" << endl;
		for (int i = 0; i < px.size()/svc.getD(); ++i) {
			cout << i << ". " << svc.predict(&px[i*svc.getD()]) << endl;
		}
	}
}

template<typename SVCT>
void do_main(
	const vector<double>& x, const vector<int>& y,
	const vector<double>& tx, const vector<int>& ty,
	const vector<double>& px, const double C, SVCT&& svc
) {
	using SVCT2 = typename remove_reference<SVCT>::type;
	using KT = typename SVCT2::kernel_type;
	if constexpr (is_same<OneAgainstAllSVC<KT>, SVCT2>::value) {
		cout << "One-against-all classification" << endl;
		auto start_t = chrono::system_clock::now();
		svc.train(x, y, C,
			[&start_t](int i) {
				cout << "Training " << i << " vs. all" << endl;
				start_t = chrono::system_clock::now();
			},
			[start_t](unsigned int nsvs, unsigned int iters) {
				auto end_t = chrono::system_clock::now();
				chrono::duration<double> elapsed = end_t-start_t;
				cout << "#SVs: " << nsvs << "(" << iters << " iterations in " << elapsed.count() << "s)" << endl;
			}
		);
		do_test_predict(tx, ty, px, svc);
	}
	if constexpr (is_same<OneAgainstOneSVC<KT>, SVCT2>::value) {
		cout << "One-against-one classification" << endl;
		auto start_t = chrono::system_clock::now();
		svc.train(x, y, C,
			[&start_t](int i, int j, size_t psize) {
				cout << "Training " << i << " vs. " << j << " (problem size: " << psize << ")" << endl;
				start_t = chrono::system_clock::now();
			},
			[start_t](unsigned int nsvs, double b, unsigned int iters) {
				auto end_t = chrono::system_clock::now();
				chrono::duration<double> elapsed = end_t-start_t;
				cout << "#SVs: " << nsvs << " B: " << b << " (" << iters << " iterations in " << elapsed.count() << "s)" << endl;
			}
		);
		do_test_predict(tx, ty, px, svc);
	}
}

template<typename T>
void do_read_data(vector<T>& out, const string& path, const string& option_fmt_key, const optparse::Values& options) {
	// if option is set, use the format specified by the user. autodetect otherwise
	if (options.is_set(option_fmt_key))
		read_dataset(out, path, format_name_to_io_format(options[option_fmt_key]));
	else
		read_dataset(out, path);
}

int main(int argc, char** argv) {
	#ifdef __CUDACC__
	int nCudaDevices;
	cudaGetDeviceCount(&nCudaDevices);
	cout << "Device count: " << nCudaDevices << endl;
	for (int i = 0; i < nCudaDevices; i++) {
		cudaDeviceProp prop;
		cudaGetDeviceProperties(&prop, i);
		cout << "Device #" << i << endl;
		cout << "\tName: " << prop.name << endl;
		cout << "\tMemory Clock (KHz): " << prop.memoryClockRate << endl;
		cout << "\tMemory Bus Width (bits): " << prop.memoryBusWidth << endl;
	}
	#endif

	const string KERNEL_OPTION_LINEAR = "linear";
	const string KERNEL_OPTION_POLYNOMIAL = "poly";
	const string KERNEL_OPTION_RBF = "rbf";
	const vector<string> KERNEL_OPTIONS {KERNEL_OPTION_LINEAR, KERNEL_OPTION_POLYNOMIAL, KERNEL_OPTION_RBF};

	const string NORMALIZATION_OPTION_ZO = "0-1";
	const string NORMALIZATION_OPTION_MOO = "-1-1";
	const string NORMALIZATION_OPTION_STANDARD = "standard";
	const vector<string> NORMALIZATION_OPTIONS {NORMALIZATION_OPTION_ZO, NORMALIZATION_OPTION_MOO, NORMALIZATION_OPTION_STANDARD};

	const string SVC_OPTION_1AA = "1AA";
	const string SVC_OPTION_1A1 = "1A1";
	const vector<string> SVC_OPTIONS {SVC_OPTION_1AA, SVC_OPTION_1A1};

	optparse::OptionParser parser;
	parser.add_option("--train-attributes").dest("train-attributes").help("Specifies training dataset with attributes (required)");
	parser.add_option("--train-labels").dest("train-labels").help("Specifies training dataset with labels (required)");
	parser.add_option("--train-format").dest("train-format")
		.type("choice").choices(begin(IO_FORMAT_NAMES), end(IO_FORMAT_NAMES))
		.help("Specifies training dataset format (" + join_str("|", IO_FORMAT_NAMES) + ")");
	parser.add_option("--train-n").dest("train-n").type("int")
		.help("Specifies amount of training samples to use (optional)");
	parser.add_option("--test-attributes").dest("test-attributes")
		.help("Specifies testing dataset attributes (optional)");
	parser.add_option("--test-labels").dest("test-labels")
		.help("Specifies testing dataset labels (required if testing)");
	parser.add_option("--test-format").dest("test-format")
		.type("choice").choices(begin(IO_FORMAT_NAMES), end(IO_FORMAT_NAMES))
		.help("Specifies testing dataset format (" + join_str("|", IO_FORMAT_NAMES) + ")");
	parser.add_option("--test-n").dest("test-n").type("int")
		.help("Specifies amount of testing samples to use (optional)");
	parser.add_option("--predict-attributes").dest("predict-attributes")
		.help("Specifies predict dataset with attributes (optional)");
	parser.add_option("--predict-format").dest("predict-format")
		.type("choices").choices(begin(IO_FORMAT_NAMES), end(IO_FORMAT_NAMES))
		.help(
			string("Specifies predict dataset format (") + join_str("|", IO_FORMAT_NAMES) + ")"
		);
	parser.add_option("--normalization", "-n").dest("normalization")
		.type("choice").choices(begin(NORMALIZATION_OPTIONS), end(NORMALIZATION_OPTIONS))
		.help(
			string("Specifies attribute normalization method (") + join_str("|", NORMALIZATION_OPTIONS) + ")"
		);
	parser.add_option("--cost", "-C").dest("cost").type("double")
		.help("Specifies the cost factor C");
	parser.add_option("--kernel", "-k").dest("kernel")
		.type("choice").choices(begin(KERNEL_OPTIONS), end(KERNEL_OPTIONS))
		.help(
			string("Specifies the svm kernel to use (") + join_str("|", KERNEL_OPTIONS) + ")"
		);
	parser.add_option("--gamma").dest("kernel-gamma").type("double")
		.help("Specifies the gamma factor for RBF kernel (gamma>0)");
	parser.add_option("--degree").dest("kernel-d").type("double")
		.help("Specifies the degree for polynomial kernel");
	parser.add_option("--constant").dest("kernel-c").type("double")
		.help("Specifies the c constant for polynomial kernel (set c=0 to use homogeneous)");
	parser.add_option("--SVC").dest("svc")
		.type("choice").choices(begin(SVC_OPTIONS), end(SVC_OPTIONS))
		.help(
			string("Select multiclass classification method (") + join_str("|", SVC_OPTIONS) + ")"
		);
	const optparse::Values options = parser.parse_args(argc, argv);

	double C = 1.0;
	if (options.is_set("cost"))
		C = double(options.get("cost"));
	cout << "Using cost factor C=" << C << endl;

	vector<double> x;
	vector<int> y;

	vector<double> test_x;
	vector<int> test_y;

	vector<double> predict_x;

	if (!options.is_set("train-attributes")) {
		cerr << "DATASET ERROR: No training attributes supplied" << endl;
		return 1;
	}
	if (!options.is_set("train-labels")) {
		cerr << "DATASET ERROR: No training labels supplied" << endl;
		return 1;
	}
	cout << "Reading dataset..." << endl;
	try {
		if (options.is_set("train-attributes") and options.is_set("train-labels")) {
			do_read_data(x, options["train-attributes"], "train-format", options);
			do_read_data(y, options["train-labels"],     "train-format", options);

			if (y.size() == 0) {
				cerr << "DATASET ERROR: no labels found at " + options["train-labels"] << endl;
				return 1;
			}
			else if (x.size() % y.size() != 0) {
				cerr << "DATASET ERROR: number of training attributes not divisible by number of labels" << endl;
				return 1;
			}
		}
		if (options.is_set("test-attributes") and options.is_set("test-labels")) {
			do_read_data(test_x, options["test-attributes"], "test-format", options);
			do_read_data(test_y, options["test-labels"],     "test-format", options);
			if (test_y.size() == 0) {
				cerr << "DATASET ERROR: no labels found at " + options["train-labels"] << endl;
				return 1;
			}
			else if (test_x.size() % test_y.size() != 0) {
				cerr << "DATASET ERROR: number of testing attributes not divisible by number of labels" << endl;
				return 1;
			}
			else if (test_x.size() / test_y.size() != x.size() / y.size()) {
				cerr << "DATASET ERROR: number of attributes in testing set doesn't match attributes in training set" << endl;
			}
		}
		if (options.is_set("predict-attributes")) {
			do_read_data(predict_x, options["predict-attributes"], "predict-format", options);
			if (predict_x.size() % y.size() != 0) {
				cerr << "DATASET ERROR: number of predicting attributes not divisible by number of labels" << endl;
			}
		}
	}
	catch (DatasetError e) {
		cerr << "DATASET ERROR: ";
		switch (e.code) {
			case DatasetError::INCONSISTENT_ROWS:
				cerr << "number of columns in row doesn't match with the rest of the dataset";
				break;
			case DatasetError::HEADER_MISMATCH:
				cerr << "header mismatch";
				break;
			case DatasetError::INVALID_TYPE:
				cerr << "invalid data type";
				break;
			case DatasetError::UNKNOWN_FORMAT:
				cerr << "unknown dataset format";
				break;
			default:
				cerr << "error in dataset";
				break;
		}
		if (e.filename != "") {
			cerr << " at " << e.filename;
			if (e.position != 0)
				cerr << ":" << e.position;
		}
		cerr << endl;
		return 1;
	}

	const int num_attributes = x.size() / y.size();
	if (options.is_set("train-n")) {
		int train_size = int(options.get("train-n"));
		x.resize(train_size*(x.size() / y.size()));
		y.resize(train_size);
	}
	if (options.is_set("test-n")) {
		size_t test_size = int(options.get("test-n"));
		test_x.resize(test_size*(x.size() / y.size()));
		test_y.resize(test_size);
	}

	std::set<int> y_set(begin(y), end(y));
	const int num_classes = y_set.size();
	cout << x.size() << " training datapoints divided between " << y.size() << " instances and " << num_classes << " classes" << endl;
	cout << test_x.size() << " testing datapoints" << endl;

	for (int i = 0; i < num_classes; i++) {
		if (y_set.find(i) == end(y_set)) {
			cerr << "ERROR: classes are not contiguous" << endl;
			return 1;
		}
	}

	if (options.is_set("normalization")) {
		const int num_attributes = x.size() / y.size();
		double scale_min = 0.;
		double scale_max = 0.;
		if (options["normalization"] == NORMALIZATION_OPTION_ZO) {
			cout << "Scaling attributes to [0;1] range" << endl;
			scale_min = 0.;
			scale_max = 1.;
		}
		else if (options["normalization"] == NORMALIZATION_OPTION_MOO) {
			cout << "Scaling attributes to [-1;1] range" << endl;
			scale_min = -1.;
			scale_max = 1.;
		}
		if (scale_min != scale_max) {
			for (int i = 0; i < num_attributes; ++i) {
				auto x_min = *min_element(strided_begin(x, i, num_attributes), strided_end(x, i, num_attributes));
				auto x_max = *max_element(strided_begin(x, i, num_attributes), strided_end(x, i, num_attributes));
				for (auto& dataset : ref_range(x, test_x, predict_x))
					for (double& x : strided_range(dataset, i, num_attributes))
						x = normalization_scale(x, scale_min, scale_max, x_min, x_max);
			}
		}
		else if (options["normalization"] == NORMALIZATION_OPTION_STANDARD) {
			cout << "Normalizing attributes to (0;1) distribution" << endl;
			for (int i = 0; i < num_attributes; ++i) {
				auto x_mean = mean(strided_begin(x, i, num_attributes), strided_end(x, i, num_attributes));
				auto x_stdev = stdev(strided_begin(x, i, num_attributes), strided_end(x, i, num_attributes), x_mean);
				for (auto& dataset : ref_range(x, test_x, predict_x))
					for (double& x : strided_range(dataset, i, num_attributes))
						x = normalization_standard(x, x_mean, x_stdev);
			}
		}
		else {
			cerr << "Unknown normalization option " << options["normalization"] << endl;
			return 1;
		}
	}

	string svc_selection = options.is_set("svc") ? options["svc"] : SVC_OPTION_1A1;
	string kernel_selection = options.is_set("kernel") ? options["kernel"] : KERNEL_OPTION_LINEAR;
	if (kernel_selection == KERNEL_OPTION_LINEAR) {
		cout << "Using linear kernel" << endl;
		auto k = LinearKernel();
		if (svc_selection == SVC_OPTION_1AA) {
			do_main(x, y, test_x, test_y, predict_x, C, OneAgainstAllSVC<LinearKernel>(num_classes, num_attributes, k));
		}
		else if (svc_selection == SVC_OPTION_1A1) {
			do_main(x, y, test_x, test_y, predict_x, C, OneAgainstOneSVC<LinearKernel>(num_classes, num_attributes, k));
		}
	}
	else if (kernel_selection == KERNEL_OPTION_RBF) {
		double gamma = 0.05;
		if (options.is_set("kernel-gamma"))
			gamma = double(options.get("kernel-gamma"));
		cout << "Using RBF kernel with gamma=" << gamma << endl;
		auto k = RbfKernel(gamma);
		if (svc_selection == SVC_OPTION_1AA) {
			do_main(x, y, test_x, test_y, predict_x, C, OneAgainstAllSVC<RbfKernel>(num_classes, num_attributes, k));
		}
		else if (svc_selection == SVC_OPTION_1A1) {
			do_main(x, y, test_x, test_y, predict_x, C, OneAgainstOneSVC<RbfKernel>(num_classes, num_attributes, k));
		}
	}
	else if (kernel_selection == KERNEL_OPTION_POLYNOMIAL) {
		double d = 2.0;
		double c = 0.0;
		if (options.is_set("kernel-d"))
			d = double(options.get("kernel-d"));
		if (options.is_set("kernel-c"))
			c = double(options.get("kernel-c"));
		cout << "Using Poly kernel with d=" << d << ", c=" << c << endl;
		auto k = PolynomialKernel(d,C);
		if (svc_selection == SVC_OPTION_1AA) {
			do_main(x, y, test_x, test_y, predict_x, C, OneAgainstAllSVC<PolynomialKernel>(num_classes, num_attributes, k));
		}
		else if (svc_selection == SVC_OPTION_1A1) {
			do_main(x, y, test_x, test_y, predict_x, C, OneAgainstOneSVC<PolynomialKernel>(num_classes, num_attributes, k));
		}
	}
}
