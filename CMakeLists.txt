
cmake_minimum_required (VERSION 3.0)
project (psvm LANGUAGES CXX CUDA)

add_library    (io_formats src/io_formats.cpp)
add_executable (cusvm src/main.cpp)
add_executable (svm src/main.cpp)

target_link_libraries (svm io_formats)
target_link_libraries (cusvm io_formats)

set_target_properties (svm PROPERTIES CXX_STANDARD 14 RUNTIME_OUTPUT_DIRECTORY bin)
set_target_properties (cusvm PROPERTIES LINKER_LANGUAGE CUDA CUDA_STANDARD 14 RUNTIME_OUTPUT_DIRECTORY cubin)
