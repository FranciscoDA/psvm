
CPP=g++
CUDACC=nvcc
CPPFLAGS=-std=c++11
CUDACCFLAGS=$(CPPFLAGS) --expt-extended-lambda -x cu

DEPS=svm.h

all: svm cusvm

svm.o: main.cpp sequential_solvers.cpp $(DEPS)
	$(CPP) $< $(CPPFLAGS) -c -o $@

svm: svm.o
	$(CPP) $< $(CPPFLAGS) -o $@

cusvm: main.cpp cuda_solvers.cu $(DEPS)
	$(CUDACC) $< $(CUDACCFLAGS) -o $@

clean:
	rm -f svm.o
	rm -f svm
	rm -f cusvm
