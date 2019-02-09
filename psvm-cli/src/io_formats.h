
#ifndef _IO_FORMATS_H_
#define _IO_FORMATS_H_

#include <vector>
#include <string>

struct DatasetError {
	enum ErrorCode {
		NONE,
		INCONSISTENT_ROWS,
		HEADER_MISMATCH,
		INVALID_TYPE,
		UNKNOWN_FORMAT
	};

	const ErrorCode code;
	const size_t position;
	const std::string filename;

	DatasetError(ErrorCode code, size_t position=0, const std::string& filename="");
};

enum class IO_FORMAT {
	NONE,
	CSV,
	IDX
};
const std::vector<std::string> IO_FORMAT_NAMES {"csv", "idx"};

IO_FORMAT format_name_to_io_format(const std::string& name);
const std::string& io_format_to_format_name(IO_FORMAT fmt);

template<typename T>
void read_dataset(std::vector<T>& x, const std::string& path, IO_FORMAT fmt);
template<typename T>
void read_dataset(std::vector<T>& x, const std::string& path);

#endif
