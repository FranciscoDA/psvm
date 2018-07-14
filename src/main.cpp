
#include "svm.h"

#ifdef __CUDACC__
#include "cuda_solvers.cu"
#else
#include "sequential_solvers.h"
#endif

#include <cmath>
#include <chrono>
#include <vector>
#include <string>
#include <iostream>
#include <set>
#include <algorithm>

#include "optparse/optparse.h"

#include "classifier.h"
#include "io_formats.h"
#include "normalization.h"

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

template <typename T>
void do_read_data(const string& fn, vector<T>& out, string user_fmt_input) {
	IO_FORMAT fmt = IO_FORMAT::NONE;
	size_t ext_pos = fn.rfind(".");
	if (ext_pos != string::npos) {
		fmt = format_name_to_io_format(fn.substr(ext_pos+1));
	}
	if (user_fmt_input != "") {
		fmt = format_name_to_io_format(user_fmt_input);
	}
	if (fmt != IO_FORMAT::NONE) {
		cout << "Reading " << io_format_to_format_name(fmt) << " data from " << fn << endl;
		read_dataset(out, fn, fmt);
	}
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

	const vector<string> KERNEL_OPTIONS {"linear", "poly", "rbf"};
	const vector<string> NORMALIZATION_OPTIONS {"0-1", "standard", "-1-1"};
	const string SVC_OPTION_1AA = "1AA";
	const string SVC_OPTION_1A1 = "1A1";
	const vector<string> SVC_OPTIONS {SVC_OPTION_1AA, SVC_OPTION_1A1};
	optparse::OptionParser parser;
	parser.add_option("--train-attributes").dest("train-attributes").help("Specifies training dataset with attributes (required)");
	parser.add_option("--train-labels").dest("train-labels").help("Specifies training dataset with labels (required)");
	parser.add_option("--train-format").dest("train-format")
		.type("choice").choices(begin(IO_FORMAT_NAMES), end(IO_FORMAT_NAMES))
		.help("Specifies training dataset format (csv|idx)");
	parser.add_option("--train-n").dest("train-n").type("int")
		.help("Specifies amount of training samples to use (optional)");
	parser.add_option("--test-attributes").dest("test-attributes")
		.help("Specifies testing dataset attributes (optional)");
	parser.add_option("--test-labels").dest("test-labels")
		.help("Specifies testing dataset labels (required if testing)");
	parser.add_option("--test-format").dest("test-format")
		.type("choice").choices(begin(IO_FORMAT_NAMES), end(IO_FORMAT_NAMES))
		.help("Specifies testing dataset format (csv|idx) (required if testing)");
	parser.add_option("--test-n").dest("test-n").type("int")
		.help("Specifies amount of testing samples to use (optional)");
	parser.add_option("--normalization").dest("normalization")
		.type("choice").choices(begin(NORMALIZATION_OPTIONS), end(NORMALIZATION_OPTIONS))
		.help("Specifies attribute normalization method (0-1|standard|-1-1)");
	parser.add_option("--cost").dest("cost").type("double")
		.help("Specifies the cost factor C");
	parser.add_option("--kernel").dest("kernel")
		.type("choice").choices(begin(KERNEL_OPTIONS), end(KERNEL_OPTIONS))
		.help("Specifies the svm kernel to use (linear|poly|rbf)");
	parser.add_option("--gamma").dest("kernel-gamma").type("double")
		.help("Specifies the gamma factor for RBF kernel (gamma>0)");
	parser.add_option("--degree").dest("kernel-d").type("double")
		.help("Specifies the degree for polynomial kernel");
	parser.add_option("--constant").dest("kernel-c").type("double")
		.help("Specifies the c constant for polynomial kernel (set c=0 to use homogeneous)");
	parser.add_option("--predict-attributes").dest("predict-attributes")
		.help("Specifies predict dataset with attributes (optional)");
	parser.add_option("--predict-format").dest("predict-format")
		.type("choices").choices(begin(IO_FORMAT_NAMES), end(IO_FORMAT_NAMES))
		.help("Specifies predict dataset format (csv|idx)");
	parser.add_option("--SVC").dest("svc")
		.type("choice").choices(begin(SVC_OPTIONS), end(SVC_OPTIONS))
		.help("Select multiclass classification method (1AA|1A1)");
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

	cout << "Reading dataset..." << endl;
	try {
		if (options.is_set("train-attributes") and options.is_set("train-labels")) {
			do_read_data(options["train-attributes"], x, options.is_set("train-format") ? options["train-format"] : "");
			do_read_data(options["train-labels"],     y, options.is_set("train-format") ? options["train-format"] : "");
		}
		if (options.is_set("test-attributes") and options.is_set("test-labels")) {
			do_read_data(options["test-attributes"], test_x, options.is_set("test-format") ? options["test-format"] : "");
			do_read_data(options["test-labels"],     test_y, options.is_set("test-format") ? options["test-format"] : "");
		}
		if (options.is_set("predict-attributes")) {
			do_read_data(options["predict-attributes"], predict_x, options.is_set("predict-format") ? options["predict-format"] : "");
		}
	}
	catch (DatasetError e) {
		cerr << "ERROR: ";
		switch (e.code) {
			case DatasetError::INCONSISTENT_D:
				cerr << "number of attributes in example don't match with the rest of the dataset";
				break;
			case DatasetError::INVALID_Y:
				cerr << "invalid class value";
				break;
			case DatasetError::HEADER_MISMATCH:
				cerr << "header mismatch";
				break;
			case DatasetError::INVALID_TYPE:
				cerr << "invalid type";
				break;
			default:
				cerr << "error in dataset";
				break;
		}
		cerr << " at line " << e.position << endl;
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
		int num_attributes = x.size() / y.size();
		vector<double> attribute_max (num_attributes, -std::numeric_limits<double>::infinity());
		vector<double> attribute_min (num_attributes, std::numeric_limits<double>::infinity());
		vector<double> attribute_mean (num_attributes, 0);
		vector<double> attribute_stdev (num_attributes, 0);

		for (int j = 0; j < num_attributes; ++j) {
			for (int i = 0; i < y.size(); ++i) {
				double v = x[i*num_attributes+j];
				attribute_max[j] = max(attribute_max[j], v);
				attribute_min[j] = min(attribute_min[j], v);
				attribute_mean[j] += v / double(y.size());
			}
			for (int i = 0; i < y.size(); ++i) {
				double v = x[i*num_attributes+j];
				attribute_stdev[j] += pow(v-attribute_mean[j], 2.0) / double(y.size());
			}
			attribute_stdev[j] = sqrt(attribute_stdev[j]);
		}
		if (options["normalization"] == "0-1") {
			cout << "Normalizing attributes to [0;1] range" << endl;
			normalization_scale(x, attribute_min, attribute_max, 0., 1.);
			normalization_scale(test_x, attribute_min, attribute_max, 0., 1.);
			normalization_scale(predict_x, attribute_min, attribute_max, 0., 1.);
		}
		else if (options["normalization"] == "-1-1") {
			cout << "Normalizing attributes to [-1;1] range" << endl;
			normalization_scale(x, attribute_min, attribute_max, -1., 1.);
			normalization_scale(test_x, attribute_min, attribute_max, -1., 1.);
			normalization_scale(predict_x, attribute_min, attribute_max, -1., 1.);
		}
		else if (options["normalization"] == "standard") {
			cout << "Normalizing attributes to (0;1) normal distribution" << endl;
			normalization_standard(x, attribute_mean, attribute_stdev);
			normalization_standard(test_x, attribute_mean, attribute_stdev);
			normalization_standard(predict_x, attribute_mean, attribute_stdev);
		}
		else {
			cerr << "Unknown normalization option " << options["normalization"] << endl;
			return 1;
		}
	}

	string svc_selection = options.is_set("svc") ? options["svc"] : SVC_OPTION_1A1;
	if (options.is_set("kernel")) {
		if (options["kernel"] == "linear") {
			cout << "Using linear kernel" << endl;
			auto k = LinearKernel();
			if (svc_selection == SVC_OPTION_1AA) {
				auto svc = OneAgainstAllSVC<LinearKernel>(num_classes, num_attributes, k);
				do_main(x, y, test_x, test_y, predict_x, C, OneAgainstAllSVC<LinearKernel>(num_classes, num_attributes, k));
			}
			else if (svc_selection == SVC_OPTION_1A1) {
				auto svc = OneAgainstOneSVC<LinearKernel>(num_classes, num_attributes, k);
				do_main(x, y, test_x, test_y, predict_x, C, svc);
			}
		}
		else if (options["kernel"] == "rbf") {
			double gamma = 0.05;
			if (options.is_set("kernel-gamma"))
				gamma = double(options.get("kernel-gamma"));
			cout << "Using RBF kernel with gamma=" << gamma << endl;
			auto k = RbfKernel(gamma);
			if (svc_selection == SVC_OPTION_1AA) {
				auto svc = OneAgainstAllSVC<RbfKernel>(num_classes, num_attributes, k);
				do_main(x, y, test_x, test_y, predict_x, C, svc);
			}
			else if (svc_selection == SVC_OPTION_1A1) {
				auto svc = OneAgainstOneSVC<RbfKernel>(num_classes, num_attributes, k);
				do_main(x, y, test_x, test_y, predict_x, C, svc);
			}
		}
		else if (options["kernel"] == "poly") {
			double d = 2.0;
			double c = 0.0;
			if (options.is_set("kernel-d"))
				d = double(options.get("kernel-d"));
			if (options.is_set("kernel-c"))
				c = double(options.get("kernel-c"));
			cout << "Using Poly kernel with d=" << d << ", c=" << c << endl;
			auto k = PolynomialKernel(d,C);
			if (svc_selection == SVC_OPTION_1AA) {
				auto svc = OneAgainstAllSVC<PolynomialKernel>(num_classes, num_attributes, k);
				do_main(x, y, test_x, test_y, predict_x, C, svc);
			}
			else if (svc_selection == SVC_OPTION_1A1) {
				auto svc = OneAgainstOneSVC<PolynomialKernel>(num_classes, num_attributes, k);
				do_main(x, y, test_x, test_y, predict_x, C, svc);
			}
		}
	}
}
