
CPP=g++
CUDACC=nvcc
CPPFLAGS=-std=c++11
CUDACCFLAGS=$(CPPFLAGS) --expt-extended-lambda -x cu

DEPS=svm.h sequential_solvers.cpp cuda_solvers.cu

all: svm cusvm

svm: main.cpp $(DEPS)
	$(CPP) $< $(CPPFLAGS) -o $@

cusvm: main.cpp $(DEPS)
	$(CUDACC) $< $(CUDACCFLAGS) -o $@

clean:
	rm -f svm
	rm -f cusvm
