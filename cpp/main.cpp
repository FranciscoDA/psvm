
#include "svm.h"

#ifdef __CUDACC__
#include "cuda_solvers.cu"
#else
#include "sequential_solvers.cpp"
#endif

#include <cmath>
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

void readCSV(vector<double>& x, vector<double>& y, string path) {
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
		y.push_back(val);
		n++;
		if (val != 1.0 and val != -1.0)
		throw DatasetError {DatasetError::ErrorCode::INVALID_Y, n};
		if (this_d != d) {
			if (n == 1)
			d = this_d;
			else
			throw DatasetError {DatasetError::ErrorCode::INCONSISTENT_D, n};
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
		switch(type){
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

	vector<double> x;
	vector<double> y;

	vector<double> test_x;
	vector<double> test_y;

	cout << "Reading dataset..." << endl;
	try {
		//readCSV(x, y, "dataset.csv");
		readIDX(x, "mnist/train-images.idx3-ubyte");
		readIDX(y, "mnist/train-labels.idx1-ubyte");
		readIDX(test_x, "mnist/t10k-images.idx3-ubyte");
		readIDX(test_y, "mnist/t10k-labels.idx1-ubyte");
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
	size_t train_size = 5000;
	size_t test_size = 500;
	x.resize(train_size*(x.size() / y.size()));
	y.resize(train_size);
	test_x.resize(test_size*(x.size() / y.size()));
	test_y.resize(test_size);
	cout << x.size() << " datapoints divided between " << y.size() << " instances" << endl;

	set<double> classes(begin(y), end(y));

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
	}
}
