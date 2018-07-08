
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

enum class IO_FORMAT {
	NONE,
	CSV,
	IDX
};
const std::vector<std::string> IO_FORMAT_NAMES {"csv", "idx"};

IO_FORMAT format_name_to_io_format(std::string name);
const std::string& io_format_to_format_name(IO_FORMAT fmt);

template<typename T>
void read_dataset(std::vector<T>& x, const std::string& path, IO_FORMAT fmt);

/*template<typename T>
void readCSV(std::vector<T>& x, std::string path);

template<typename T>
void readIDX(std::vector<T>& x, std::string path);*/

#endif
