
#ifndef _IO_FORMATS_H_
#define _IO_FORMATS_H_

#include <vector>
#include <string>

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

template<typename T>
void readCSV(std::vector<T>& x, std::string path);

template<typename T>
void readIDX(std::vector<T>& x, std::string path);

#endif
