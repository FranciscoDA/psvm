
#include "svm.h"

#ifdef __CUDACC__
#include "cuda_solvers.cu"
#else
#include "sequential_solvers.h"
#endif

#include "classifier.h"
#include "io_formats.h"
#include "optparse/optparse.h"

#include <cmath>
#include <chrono>
#include <vector>
#include <string>
#include <iostream>
#include <set>

using namespace std;

template<KERNEL_TYPE KT, SVC_TYPE SVCT>
void do_test_predict(
	const vector<double>& tx, const vector<int>& ty,
	const vector<double>& px, const SVC<KT, SVCT>& svc
) {
	if (ty.size() > 0) {
		unsigned int hits = testSVC(svc, tx, ty);
		double accuracy = double(hits)/double(ty.size());
		cout << "Model accuracy: " << hits << "/" << ty.size() << " = " << accuracy << endl;
	}
	if (px.size() > 0) {
		cout << "Predictions:" << endl;
		for (int i = 0; i < px.size()/svc.getD(); ++i) {
			cout << i << ". " << svc.predict(&px[i*svc.getD()]) << endl;
		}
	}
}

template<KERNEL_TYPE KT>
void do_main(
	const vector<double>& x, const vector<int>& y,
	const vector<double>& tx, const vector<int>& ty,
	const vector<double>& px, const double C, SVC_TYPE svctype,
	const Kernel<KT>& kernel, const int num_classes
) {
	switch (svctype) {
		case SVC_TYPE::OAA: {
			cout << "One-against-all classification" << endl;
			SVC<KT, SVC_TYPE::OAA> svc(num_classes, x.size() / y.size(), kernel);
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
			break;
		}
		case SVC_TYPE::OAO: {
			cout << "One-against-one classification" << endl;
			SVC<KT, SVC_TYPE::OAO> svc(num_classes, x.size() / y.size(), kernel);
			auto start_t = chrono::system_clock::now();
			svc.train(x, y, C,
				[&start_t](int i, int j) {
					cout << "Training " << i << " vs. " << j << endl;
					start_t = chrono::system_clock::now();
				},
				[start_t](unsigned int nsvs, unsigned int iters) {
					auto end_t = chrono::system_clock::now();
					chrono::duration<double> elapsed = end_t-start_t;
					cout << "#SVs: " << nsvs << "(" << iters << " iterations in " << elapsed.count() << "s)" << endl;
				}
			);
			do_test_predict(tx, ty, px, svc);
			break;
		}
		case SVC_TYPE::TWOCLASS: {
			cout << "Two class classification" << endl;
			SVC<KT, SVC_TYPE::TWOCLASS> svc(x.size() / y.size(), kernel);
			auto start_t = chrono::system_clock::now();
			svc.train(x, y, C,
				[](int i, int j) { cout << "Training " << i << " vs. " << j << endl; },
				[&start_t](unsigned int nsvs, unsigned int iters) {
					auto end_t = chrono::system_clock::now();
					chrono::duration<double> elapsed = end_t-start_t;
					cout << "#SVs: " << nsvs << "(" << iters << " iterations in " << elapsed.count() << "s)" << endl;
				}
			);
			do_test_predict(tx, ty, px, svc);
			break;
		}
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

	const vector<string> FORMAT_OPTIONS {"csv", "idx"};
	const vector<string> KERNEL_OPTIONS {"linear", "poly", "rbf"};
	optparse::OptionParser parser;
	parser.add_option("--train-attributes").dest("train-attributes").help("Specifies training dataset with attributes (required)");
	parser.add_option("--train-labels").dest("train-labels").help("Specifies training dataset with labels (required)");
	parser.add_option("--train-format").dest("train-format")
		.type("choice").choices(begin(FORMAT_OPTIONS), end(FORMAT_OPTIONS))
		.help("Specifies training dataset format (csv|idx)");
	parser.add_option("--train-n").dest("train-n").type("int").help("Specifies amount of training samples to use (optional)");

	parser.add_option("--test-attributes").dest("test-attributes").help("Specifies testing dataset attributes (optional)");
	parser.add_option("--test-labels").dest("test-labels").help("Specifies testing dataset labels (required if testing)");
	parser.add_option("--test-format").dest("test-format")
		.type("choice").choices(begin(FORMAT_OPTIONS), end(FORMAT_OPTIONS))
		.help("Specifies testing dataset format (csv|idx) (required if testing)");
	parser.add_option("--test-n").dest("test-n").type("int").help("Specifies amount of testing samples to use (optional)");

	parser.add_option("--normalize-zero-one").action("store_true").dest("normalize-zero-one").help("Preprocesses data to fit [0;1] range");
	parser.add_option("--cost").dest("cost").type("double").help("Specifies the cost factor C");
	parser.add_option("--kernel").dest("kernel")
		.type("choice").choices(begin(KERNEL_OPTIONS), end(KERNEL_OPTIONS))
		.help("Specifies the svm kernel to use (linear|poly|rbf)");
	parser.add_option("--gamma").dest("kernel-gamma").type("double").help("Specifies the gamma factor for RBF kernel (gamma>0)");
	parser.add_option("--degree").dest("kernel-d").type("double").help("Specifies the degree for polynomial kernel");
	parser.add_option("--constant").dest("kernel-c").type("double").help("Specifies the c constant for polynomial kernel (set c=0 to use homogeneous)");

	parser.add_option("--predict-attributes").dest("predict-attributes").help("Specifies predict dataset with attributes (optional)");
	parser.add_option("--predict-format").dest("predict-format")
		.type("choices").choices(begin(FORMAT_OPTIONS), end(FORMAT_OPTIONS))
		.help("Specifies predict dataset format (csv|idx)");

	parser.add_option("--1AA").action("store_true").help("Train and classify using one-against-all algorithm");
	parser.add_option("--1A1").action("store_true").help("Train and classify using one-against-one algorithm");
	parser.add_option("--TWOCLASS").action("store_true").help("Train and classify using two-class algorithm");
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
		if (options.is_set("train-attributes") and options.is_set("train-labels") and options.is_set("train-format")) {
			cout << "Reading train data from " << options["train-attributes"] << " and " << options["train-labels"] << endl;
			if (options["train-format"] == "csv") {
				readCSV(x, options["train-attributes"]);
				readCSV(y, options["train-labels"]);
			}
			else if (options["train-format"] == "idx") {
				readIDX(x, options["train-attributes"]);
				readIDX(y, options["train-labels"]);
			}
		}
		if (options.is_set("test-attributes") and options.is_set("test-labels") and options.is_set("test-format")) {
			cout << "Reading test data from " << options["test-attributes"] << " and " << options["test-labels"] << endl;
			if (options["test-format"] == "csv") {
				readCSV(test_x, options["test-attributes"]);
				readCSV(test_y, options["test-labels"]);
			}
			else if (options["test-format"] == "idx") {
				readIDX(test_x, options["test-attributes"]);
				readIDX(test_y, options["test-labels"]);
			}
		}
		if (options.is_set("predict-attributes") and options.is_set("predict-format")) {
			cout << "Reading predict data from " << options["predict-attributes"] << endl;
			if (options["predict-format"] == "csv") {
				readCSV(predict_x, options["predict-attributes"]);
			}
			else if (options["predict-format"] == "idx") {
				readIDX(predict_x, options["predict-attributes"]);
			}
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
	int num_classes = y_set.size();
	cout << x.size() << " datapoints divided between " << y.size() << " instances and " << num_classes << " classes" << endl;
	for (int i = 0; i < num_classes; i++) {
		if (y_set.find(i) == end(y_set)) {
			cerr << "ERROR: classes are not contiguous" << endl;
		}
	}

	if (options.is_set("normalize-zero-one")) {
		double _min = x[0];
		double _max = x[0];
		for (double x_i : x) {
			if (x_i < _min) _min = x_i;
			if (_max < x_i) _max = x_i;
		}
		cout << "Normalizing training set from [" << _min << ";" << _max << "] to [0;1]" << endl;
		for(double& x_i : x) x_i = (x_i - _min) / (_max - _min);
		if (test_y.size() > 0) {
			cout << "Normalizing test set from [" << _min << ";" << _max << "] to [0;1]" << endl;
			for(double& x_i : test_x) x_i = (x_i - _min) / (_max - _min);
		}
		if (predict_x.size() > 0) {
			cout << "Normalizing predict set from [" << _min << ";" << _max << "] to [0;1]" << endl;
			for (double& x_i : predict_x) x_i = (x_i - _min) / (_max-_min);
		}
	}

	SVC_TYPE svctype;
	if (options.is_set("1A1")) {
		svctype = SVC_TYPE::OAO;
	}
	else if (options.is_set("1AA")) {
		svctype = SVC_TYPE::OAA;
	}
	else if (options.is_set("TWOCLASS")) {
		svctype = SVC_TYPE::TWOCLASS;
	}

	if (options.is_set("kernel")) {
		if (options["kernel"] == "linear") {
			cout << "Using linear kernel" << endl;
			do_main<KERNEL_TYPE::LINEAR>(x, y, test_x, test_y, predict_x, C, svctype, Kernel<KERNEL_TYPE::LINEAR>(), num_classes);
		}
		else if (options["kernel"] == "rbf") {
			double gamma = 0.05;
			if (options.is_set("kernel-gamma"))
				gamma = double(options.get("kernel-gamma"));
			cout << "Using RBF kernel with gamma=" << gamma << endl;
			do_main<KERNEL_TYPE::RBF>(x, y, test_x, test_y, predict_x, C, svctype, Kernel<KERNEL_TYPE::RBF>(gamma), num_classes);
		}
		else if (options["kernel"] == "poly") {
			double d = 2.0;
			double c = 0.0;
			if (options.is_set("kernel-d"))
				d = double(options.get("kernel-d"));
			if (options.is_set("kernel-c"))
				c = double(options.get("kernel-c"));
			cout << "Using Poly kernel with d=" << d << ", c=" << c << endl;
			do_main<KERNEL_TYPE::POLYNOMIAL>(x, y, test_x, test_y, predict_x, C, svctype, Kernel<KERNEL_TYPE::POLYNOMIAL>(d, c), num_classes);
		}
	}
}
