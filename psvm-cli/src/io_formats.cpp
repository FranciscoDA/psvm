
#include <cmath>
#include <cctype>
#include <vector>
#include <fstream>
#include <sstream>
#include <string>
#include <algorithm>
#include <type_traits>

#include "io_formats.h"

DatasetError::DatasetError(DatasetError::ErrorCode code, size_t position, const std::string& filename)
: code(code), position(position), filename(filename) {
}

IO_FORMAT format_name_to_io_format(const std::string& name) {
	auto it = std::find_if(IO_FORMAT_NAMES.cbegin(), IO_FORMAT_NAMES.cend(), [&name](const auto& v) {
		return std::equal(v.cbegin(), v.cend(), name.cbegin(), name.cend(), [](int a, int b) { return tolower(a) == tolower(b); });
	});
	if (it != end(IO_FORMAT_NAMES)) {
		return static_cast<IO_FORMAT>(std::distance(begin(IO_FORMAT_NAMES), it)+1);
	}
	return IO_FORMAT::NONE;
}
const std::string& io_format_to_format_name(IO_FORMAT fmt) {
	int i = static_cast<int>(fmt)-1;
	return IO_FORMAT_NAMES[i];
}

template<typename T>
void read_CSV(std::vector<T>& x, const std::string& path) {
	size_t n = 0;
	size_t d = 0;
	std::fstream dataset(path, std::ios_base::in);
	while (!dataset.eof()) {
		std::string line;
		dataset >> line;
		if (!line.length())
			continue; // empty line
		std::stringstream ss(line);
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

template <typename InType, typename OutType>
typename std::enable_if<std::is_convertible<InType, OutType>::value>::type castIDX(InType inValue, OutType& outValue) {
	outValue = static_cast<OutType>(inValue);
}
template <typename InType>
void castIDX(InType inValue, std::string& outValue) {
	outValue = std::to_string(inValue);
}
template <typename OutType>
typename std::enable_if<std::is_integral<OutType>::value, OutType>::type valueIDX(std::istream& stream) {
	OutType v = 0;
	for (size_t i = 0; i < sizeof(OutType); ++i)
		v = (v << 8) + stream.get();
	return v;
}
template<typename OutType>
typename std::enable_if<std::is_floating_point<OutType>::value, OutType>::type valueIDX(std::istream& stream) {
	OutType v;
	stream.read(reinterpret_cast<char*>(&v), sizeof(v));
	return v;
}

template<typename T>
void read_IDX(std::vector<T>& x, const std::string& path) {
	std::fstream dataset(path, std::ios_base::in | std::ios_base::binary);
	for (int i = 0; i < 2; ++i)
		if (dataset.get() != 0)
			throw DatasetError {DatasetError::ErrorCode::HEADER_MISMATCH, 0, path};
	int type = dataset.get();
	size_t d = dataset.get();
	size_t dn = 1;
	for (int i = 0; i < d; ++i) {
		size_t s = valueIDX<int>(dataset);
		dn *= s;
	}
	x.resize(dn);
	for (size_t i = 0; i < dn; ++i) {
		switch(type) {
			case 0x08: // unsigned byte
				castIDX(valueIDX<unsigned char>(dataset), x[i]);
				break;
			case 0x09:
				// TODO: handle signed bytes
			case 0x0B: // short
				castIDX(valueIDX<short>(dataset), x[i]);
				break;
			case 0x0C: // int
				castIDX(valueIDX<int>(dataset), x[i]);
				break;
			case 0x0D: // float
				castIDX(valueIDX<float>(dataset), x[i]);
				break;
			case 0x0E: // double
				castIDX(valueIDX<double>(dataset), x[i]);
				break;
			default:
				throw DatasetError {DatasetError::ErrorCode::INVALID_TYPE, i, path};
		}
	}
}

template<typename T>
void read_dataset(std::vector<T>& x, const std::string& path, IO_FORMAT fmt) {
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
void read_dataset(std::vector<T>& x, const std::string& path) {
	// deduce format from file extension
	IO_FORMAT fmt = IO_FORMAT::NONE;
	size_t ext_pos = path.rfind(".");
	if (ext_pos != std::string::npos)
		fmt = format_name_to_io_format(path.substr(ext_pos+1));
	read_dataset(x, path, fmt);
}
template void read_dataset(std::vector<double>& x, const std::string& path);
template void read_dataset(std::vector<std::string>& x, const std::string& path);
template void read_dataset(std::vector<int>& x, const std::string& path);
