
#include <cmath>
#include <vector>
#include <fstream>
#include <sstream>
#include <string>
#include "io_formats.h"

using namespace std;

int readBinaryIntBE(istream& x, size_t len) {
	int v = 0;
	for (; len > 0; --len) v = v*256 + x.get();
	return v;
}

template<typename T>
void readCSV(vector<T>& x, string path) {
	size_t n = 0;
	size_t d = 0;
	fstream dataset(path, ios_base::in);
	while (!dataset.eof()) {
		string line;
		dataset >> line;
		if (!line.length()) continue; // empty line
		stringstream ss(line);
		T val;
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
template void readCSV(vector<double>& x, string path);
template void readCSV(vector<string>& x, string path);

template <typename T>
void readIDX(vector<T>& x, string path) {
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
				x.push_back(T(readBinaryIntBE(dataset, 1)));
				break;
			case 0x0B: // short
				x.push_back(T(readBinaryIntBE(dataset, 2)));
				break;
			case 0x0C:
				x.push_back(T(readBinaryIntBE(dataset, 4)));
				break;
			case 0x0D:
				float fl;
				dataset >> fl;
				x.push_back(T(fl));
				break;
			case 0x0E:
				double db;
				dataset >> db;
				x.push_back(T(db));
				break;
			default:
				throw DatasetError {DatasetError::ErrorCode::INVALID_TYPE, i};
		}
	}
}
template void readIDX(vector<double>& x, string path);

template<>
void readIDX<string> (vector<string>& x, string path) {
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
				x.push_back(to_string(readBinaryIntBE(dataset, 1)));
				break;
			case 0x0B: // short
				x.push_back(to_string(readBinaryIntBE(dataset, 2)));
				break;
			case 0x0C:
				x.push_back(to_string(readBinaryIntBE(dataset, 4)));
				break;
			case 0x0D:
				float fl;
				dataset >> fl;
				x.push_back(to_string(fl));
				break;
			case 0x0E:
				double db;
				dataset >> db;
				x.push_back(to_string(db));
				break;
			default:
				throw DatasetError {DatasetError::ErrorCode::INVALID_TYPE, i};
		}
	}
}
