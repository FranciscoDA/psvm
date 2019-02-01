CMAKE_MINIMUM_REQUIRED(VERSION 3.8)
PROJECT(psvm-cli LANGUAGES CXX CUDA)

ADD_EXECUTABLE (svm src/main.cpp src/io_formats.cpp)
ADD_EXECUTABLE (cusvm src/main.cu src/io_formats.cpp)

TARGET_INCLUDE_DIRECTORIES(svm   PUBLIC ${CMAKE_SOURCE_DIR}/libpsvm/include BOOST_INCLUDEDIR)
TARGET_INCLUDE_DIRECTORIES(cusvm PUBLIC ${CMAKE_SOURCE_DIR}/libpsvm/include BOOST_INCLUDEDIR)

SET_TARGET_PROPERTIES(svm PROPERTIES
                      CXX_STANDARD 17
                      RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)
SET_TARGET_PROPERTIES(cusvm PROPERTIES
                      LINKER_LANGUAGE CUDA
                      CUDA_STANDARD 14
                      RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
                      CUDA_SEPARABLE_COMPILATION on
)
TARGET_LINK_LIBRARIES(svm Boost::program_options libsvm)
TARGET_LINK_LIBRARIES(cusvm Boost::program_options libcusvm)

