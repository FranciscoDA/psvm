
#include <cmath>
#include <cctype>
#include <vector>
#include <fstream>
#include <sstream>
#include <string>
#include <algorithm>
#include <type_traits>

#include "io_formats.h"

using namespace std;

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
			if (n == 1)
				d = this_d;
			else
				throw DatasetError {DatasetError::ErrorCode::INCONSISTENT_ROWS, n, path};
		}
	}
}


// class to read a single value from an idx file and cast it to the container type
template<typename OUTTYPE>
class value_IDX {
public:
	template<typename INTYPE>
	OUTTYPE next(istream& ins) {
		return cast(get<INTYPE>(ins));
	}
protected:
	template<typename INTYPE>
	OUTTYPE cast(INTYPE in) {
		return static_cast<OUTTYPE>(in);
	}
	template<typename INTYPE>
	enable_if_t<is_integral<INTYPE>::value, INTYPE> get(istream& ins) {
		INTYPE v = 0;
		for (int i = 0; i < sizeof(INTYPE); ++i)
			v = (v << 8) + ins.get();
		return v;
	}
	// special case if INTYPE is a floating point
	template<typename INTYPE>
	enable_if_t<is_floating_point<INTYPE>::value, INTYPE> get(istream& ins) {
		INTYPE v;
		ins.read(reinterpret_cast<char*>(&v), sizeof v);
		return v;
	}
};
// specialization: when converting to string use std::to_string instead of static_cast
template<>
class value_IDX<string> : value_IDX<void> {
public:
	template<typename INTYPE>
	string next(istream& ins) {
		return cast(get<INTYPE>(ins));
	}
protected:
	template<typename INTYPE>
	string cast(INTYPE in) {
		return to_string(in);
	}
};

template<typename T>
void read_IDX(vector<T>& x, string path) {
	fstream dataset(path, ios_base::in | ios_base::binary);
	for (int i = 0; i < 2; ++i)
		if (dataset.get() != 0)
			throw DatasetError {DatasetError::ErrorCode::HEADER_MISMATCH, 0, path};
	int type = dataset.get();
	size_t d = dataset.get();
	size_t dn = 1;
	for (int i = 0; i < d; ++i) {
		size_t s = value_IDX<int>().template next<int>(dataset);
		dn *= s;
	}
	x.reserve(dn);
	value_IDX<T> parser;
	for (size_t i = 0; i < dn; ++i) {
		switch(type) {
			case 0x08: // unsigned byte
				x.push_back(parser.template next<unsigned char>(dataset));
				break;
			case 0x09:
				// TODO: handle signed bytes
			case 0x0B: // short
				x.push_back(parser.template next<short>(dataset));
				break;
			case 0x0C: // int
				x.push_back(parser.template next<int>(dataset));
				break;
			case 0x0D: // float
				x.push_back(parser.template next<float>(dataset));
				break;
			case 0x0E: // double
				x.push_back(parser.template next<double>(dataset));
				break;
			default:
				throw DatasetError {DatasetError::ErrorCode::INVALID_TYPE, i, path};
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
	else {
		throw DatasetError {DatasetError::UNKNOWN_FORMAT, 0, ""};
	}
}
template<typename T>
void read_dataset(vector<T>& x, const string& path) {
	// deduce format from file extension
	IO_FORMAT fmt = IO_FORMAT::NONE;
	size_t ext_pos = path.rfind(".");
	if (ext_pos != string::npos) {
		fmt = format_name_to_io_format(path.substr(ext_pos+1));
	}
	read_dataset(x, path, fmt);
}
template void read_dataset(vector<double>& x, const string& path);
template void read_dataset(vector<string>& x, const string& path);
template void read_dataset(vector<int>& x, const string& path);
