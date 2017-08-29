
CPP=g++
CUDACC=nvcc
CPPFLAGS=-std=c++11
CUDACCFLAGS=$(CPPFLAGS) --expt-extended-lambda -x cu

DEPS=svm.h sequential_solvers.cpp cuda_solvers.cu

all: sequential parallel

sequential: main.cpp $(DEPS)
	$(CPP) $< $(CPPFLAGS) -o $@

parallel: main.cpp $(DEPS)
	$(CUDACC) $< $(CUDACCFLAGS) -o $@

clean:
	rm -f sequential
	rm -f parallel
