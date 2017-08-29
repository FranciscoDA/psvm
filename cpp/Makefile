
CPP=g++
NVCC=nvcc
CPPFLAGS=-std=c++11
NVCCFLAGS=$(CPPFLAGS) --expt-extended-lambda

DEPS=svm.h sequential_solvers.cpp cuda_solvers.cu

all: sequential parallel

sequential: main.cpp $(DEPS)
	$(CPP) $< $(CPPFLAGS) -o $@

parallel: main.cpp $(DEPS)
	$(NVCC) $< $(NVCCFLAGS) -o $@

clean:
	rm -f sequential
	rm -f parallel
