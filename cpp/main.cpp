
#include "svm.h"

#ifdef __CUDACC__
#include "cuda_solvers.cu"
#else
#include "sequential_solvers.cpp"
#endif
#include "optparse/optparse.h"

#include <cmath>
#include <ctime>
#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <set>

using namespace std;

struct DatasetError {
	enum ErrorCode {
		INCONSISTENT_D,
		INVALID_Y,
		HEADER_MISMATCH,
		INVALID_TYPE,
	};

	ErrorCode code;
	size_t position;
};

void readCSV(vector<double>& x, string path) {
	size_t n = 0;
	size_t d = 0;
	fstream dataset(path, ios_base::in);
	while (!dataset.eof()) {
		string line;
		dataset >> line;
		if (!line.length()) // empty line
		continue;
		stringstream ss(line);
		double val;
		ss >> val;
		ss.ignore(1, ',');
		size_t this_d = 1;
		while (!ss.eof()) {
			x.push_back(val);
			ss >> val;
			ss.ignore(1, ',');
			++this_d;
		}
		x.push_back(val);
		n++;
		if (this_d != d) {
			if (n == 1) d = this_d;
			else throw DatasetError {DatasetError::ErrorCode::INCONSISTENT_D, n};
		}
	}
}
int readBinaryIntBE(istream& x, size_t len) {
	int v = 0;
	for (; len > 0; --len)
	v = v*256 + x.get();
	return v;
}
void readIDX(vector<double>& x, string path) {
	fstream dataset(path, ios_base::in | ios_base::binary);
	for (int i = 0; i < 2; ++i)
		if (dataset.get() != 0)
			throw DatasetError {DatasetError::ErrorCode::HEADER_MISMATCH, 0};
	int type = dataset.get();
	size_t d = dataset.get();
	size_t dn = 1;
	for (int i = 0; i < d; ++i) {
		size_t s = readBinaryIntBE(dataset, 4);
		dn *= s;
	}
	x.reserve(dn);
	for (size_t i = 0; i < dn; ++i) {
		switch(type) {
			case 0x08: // unsigned byte
				x.push_back((double)readBinaryIntBE(dataset, 1));
				break;
			case 0x0B: // short
				x.push_back((double)readBinaryIntBE(dataset, 2));
				break;
			case 0x0C:
				x.push_back((double)readBinaryIntBE(dataset, 4));
				break;
			case 0x0D:
				float fl;
				dataset >> fl;
				x.push_back(fl);
				break;
			case 0x0E:
				double db;
				dataset >> db;
				x.push_back(db);
				break;
			default:
				throw DatasetError {DatasetError::ErrorCode::INVALID_TYPE, i};
		}
	}
}

template<typename KT, typename ...KARG>
void trainAndMaybeTest(
	const vector<double>& x, const vector<double>& y,
	bool doTest, const vector<double>& tx, const vector<double>& ty,
	double C,
	KARG... kargs
) {
	set<double> classes(begin(y), end(y));
	if (classes.size() == 2 and classes.find(1.0) != end(classes) and classes.find(-1.0) != end(classes)) {
		cout << "Two class classification (y_i in {-1, 1})" << endl;
		cout << "Training with SMO" << endl;
		SVM<KT> svm(x.size() / y.size(), KT(kargs...));
		clock_t start = clock();
		unsigned int iterations = smo(svm, x, y, 0.01, C);
		clock_t end = clock();
		double elapsed = double(end-start)/CLOCKS_PER_SEC;
		cout << "No. of SVs: " << svm.getSVAlphaY().size() << "(" << iterations << " iterations in " << elapsed << "s)" << endl;
		if (doTest) {
			unsigned int hits = test(svm, tx, ty);
			double accuracy = double(hits)/double(ty.size());
			cout << "Model accuracy: " << hits << "/" << ty.size() << " = " << accuracy << endl;
		}
	}
	else {
		cout << "Multi-class classification (one vs all) - " << classes.size() << " classes" << endl;
		vector<SVM<KT>> classifiers;
		double elapsed_final = 0.0;
		unsigned int nsv_final = 0;
		for (double label : classes) {
			cout << "Training for label=" << label << endl;
			vector<double> y1;
			y1.reserve(y.size());
			for (double y_i : y)
				y1.push_back(y_i == label ? 1.0 : -1.0);
			SVM<KT> svm(x.size() / y.size(), KT(kargs...));
			clock_t start = clock();
			unsigned int iterations = smo(svm, x, y1, 0.01, C);
			clock_t end = clock();
			double elapsed = double(end-start)/CLOCKS_PER_SEC;
			cout << "No. of SVs: " << svm.getSVAlphaY().size() << "(" << iterations << " iterations in " << elapsed << "s)" << endl;
			classifiers.push_back(svm);
			elapsed_final += elapsed;
			nsv_final += svm.getSVAlphaY().size();
		}
		cout << "Total of SVs: " << nsv_final << endl;
		cout << "Total training time: " << elapsed_final << "s" << endl;
		if (doTest) {
			unsigned int hits = test1VA(classifiers, tx, ty);
			double accuracy = double(hits)/double(ty.size());
			cout << "Model accuracy: " << hits << "/" << ty.size() << " = " << accuracy << endl;
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

	optparse::OptionParser parser;
	parser.add_option("--train-attributes").dest("train-attributes");
	parser.add_option("--train-labels").dest("train-labels");
	parser.add_option("--train-format").dest("train-format");
	parser.add_option("--train-n").dest("train-n");

	parser.add_option("--test-attributes").dest("test-attributes");
	parser.add_option("--test-labels").dest("test-labels");
	parser.add_option("--test-format").dest("test-format");
	parser.add_option("--test-n").dest("test-n");

	parser.add_option("--normalize-zero-one").action("store_true").dest("normalize-zero-one");
	parser.add_option("--cost").dest("cost");
	parser.add_option("--kernel").dest("kernel");
	parser.add_option("--gamma").dest("kernel-gamma");
	parser.add_option("--power").dest("kernel-p");
	parser.add_option("--constant").dest("kernel-c");
	const optparse::Values options = parser.parse_args(argc, argv);

	double C = 1.0;
	if (options.is_set("cost"))
	 	C = double(options.get("cost"));
	cout << "Using cost factor C=" << C << endl;

	vector<double> x;
	vector<double> y;

	bool use_test_data = false;
	vector<double> test_x;
	vector<double> test_y;

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
				use_test_data = true;
			}
			else if (options["test-format"] == "idx") {
				readIDX(test_x, options["test-attributes"]);
				readIDX(test_y, options["test-labels"]);
				use_test_data = true;
			}
		}
		//readCSV(x, y, "dataset.csv");
		//readIDX(x, "mnist/train-images.idx3-ubyte");
		//readIDX(y, "mnist/train-labels.idx1-ubyte");
		//readIDX(test_x, "mnist/t10k-images.idx3-ubyte");
		//readIDX(test_y, "mnist/t10k-labels.idx1-ubyte");
		/*for (auto i : test_x)
		cout << i << endl;*/
	}
	catch (DatasetError e) {
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
	cout << x.size() << " datapoints divided between " << y.size() << " instances" << endl;

	if (options.is_set("normalize-zero-one")) {
		/*cout << "Normalizing training set from [0;255] to [0;1]" << endl;
		for (double& x_i : x)      x_i /= 255.0;
		for (double& x_i : test_x) x_i /= 255.0;*/
		double _min = x[0];
		double _max = x[0];
		for (double x_i : x) {
			if (x_i < _min) _min = x_i;
			if (_max < x_i) _max = x_i;
		}
		cout << "Normalizing training set from [" << _min << ";" << _max << "] to [0;1]" << endl;
		for(double& x_i : x) x_i = (x_i - _min) / (_max - _min);
		if (use_test_data) {
			cout << "Normalizing test set from [" << _min << ";" << _max << "] to [0;1]" << endl;
			for(double& x_i : test_x) x_i = (x_i - _min) / (_max - _min);
		}
	}

	if (options.is_set("kernel")) {
		if (options["kernel"] == "linear") {
			cout << "Using linear kernel" << endl;
			trainAndMaybeTest<LinearKernel>(x, y, use_test_data, test_x, test_y, C);
		}
		else if (options["kernel"] == "rbf") {
			double gamma = 0.05;
			if (options.is_set("kernel-gamma"))
				gamma = double(options.get("kernel-gamma"));
			cout << "Using RBF kernel with gamma=" << gamma << endl;
			trainAndMaybeTest<RbfKernel>(x, y, use_test_data, test_x, test_y, C, -gamma);
		}
		else if (options["kernel"] == "poly") {
			double p = 2.0;
			double c = 0.0;
			if (options.is_set("kernel-p"))
				p = double(options.get("kernel-p"));
			if (options.is_set("kernel-c"))
				p = double(options.get("kernel-c"));
			cout << "Using Poly kernel with p=" << p << ", c=" << c << endl;
			trainAndMaybeTest<PolynomialKernel>(x, y, use_test_data, test_x, test_y, C, p, c);
		}
	}

	/*set<double> classes(begin(y), end(y));

	if (classes.size() == 2 and classes.find(1.0) != end(classes) and classes.find(-1.0) != end(classes)) {
		cout << "Two class classification (y_i in {-1, 1})" << endl;
		cout << "Training with SMO" << endl;
		SVM<LinearKernel> svm(x.size() / y.size(), LinearKernel());
		smo(svm, x, y, 0.01, 2.0);
		cout << "No. of SVs: " << svm.getSVAlphaY().size() << endl;
		for(auto ity = begin(svm.getSVAlphaY()); ity != end(svm.getSVAlphaY()); ++ity)
			cout << "alpha_i * y_i = " << *ity << endl;
	}
	else {
		cout << "Multi-class classification (one vs all) - " << classes.size() << " classes" << endl;
		std::vector<SVM<RbfKernel>> classifiers;

		for (double& x_i : x)      x_i /= 255.0;
		for (double& x_i : test_x) x_i /= 255.0;

		for (double label : classes) {
			cout << "Training for label=" << label << endl;
			vector<double> y1;
			y1.reserve(y.size());
			for (double y_i : y)
				y1.push_back(y_i == label ? 1.0 : -1.0);
			SVM<RbfKernel> svm(x.size() / y.size(), RbfKernel(-0.05));
			smo(svm, x, y1, 0.001, 0.6);
			cout << "No. of SVs: " << svm.getSVAlphaY().size() << endl;
			classifiers.push_back(svm);
		}
		cout << "Model accuracy: " << test1VA(classifiers, test_x, test_y) << endl;
	}*/
}
