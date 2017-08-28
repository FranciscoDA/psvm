
CPP=g++
NVCC=nvcc
CPPFLAGS=-std=c++11
NVCCFLAGS=$(CPPFLAGS) --expt-extended-lambda

DEPS=svm.h

all: sequential parallel

sequential: solvers.cpp $(DEPS)
	g++ $< $(CPPFLAGS) -o $@

parallel: cuda_solvers.cu $(DEPS)
	nvcc $< $(NVCCFLAGS) -o $@

clean:
	rm -f sequential
	rm -f parallel
