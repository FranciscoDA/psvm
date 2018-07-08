
#include <cmath>
#include <cctype>
#include <vector>
#include <fstream>
#include <sstream>
#include <string>
#include <algorithm>

#include "io_formats.h"

using namespace std;

int readBinaryIntBE(istream& x, size_t len) {
	int v = 0;
	for (; len > 0; --len) v = v*256 + x.get();
	return v;
}

IO_FORMAT format_name_to_io_format(string name) {
	transform(begin(name), end(name), begin(name), [](auto ch) { return tolower(ch); });
	auto it = find(cbegin(IO_FORMAT_NAMES), cend(IO_FORMAT_NAMES), name);
	if (it != end(IO_FORMAT_NAMES)) {
		return static_cast<IO_FORMAT>(distance(begin(IO_FORMAT_NAMES), it)+1);
	}
	return IO_FORMAT::NONE;
}
const string& io_format_to_format_name(IO_FORMAT fmt) {
	int i = static_cast<int>(fmt)-1;
	return IO_FORMAT_NAMES[i];
}

template<typename T>
void read_CSV(vector<T>& x, string path) {
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
template void read_CSV(vector<double>& x, string path);
template void read_CSV(vector<string>& x, string path);
template void read_CSV(vector<int>& x, string path);

template <typename T>
void read_IDX(vector<T>& x, string path) {
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
template void read_IDX(vector<double>& x, string path);
template void read_IDX(vector<int>& x, string path);

template<>
void read_IDX<string> (vector<string>& x, string path) {
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

template<typename T>
void read_dataset(vector<T>& x, const string& path, IO_FORMAT fmt) {
	if (fmt == IO_FORMAT::CSV) {
		read_CSV(x, path);
	}
	else if (fmt == IO_FORMAT::IDX) {
		read_IDX(x, path);
	}
}
template void read_dataset(vector<double>& x, const string& path, IO_FORMAT fmt);
template void read_dataset(vector<string>& x, const string& path, IO_FORMAT fmt);
template void read_dataset(vector<int>& x, const string& path, IO_FORMAT fmt);
