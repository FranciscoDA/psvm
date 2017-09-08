
CPP=g++
CUDACC=nvcc
CPPFLAGS=-std=c++11
CUDACCFLAGS=$(CPPFLAGS) --expt-extended-lambda -x cu
SRCDIR=src
OUTDIR=bin
DEPS=$(SRCDIR)/svm.h

all: $(OUTDIR)/svm $(OUTDIR)/cusvm

$(OUTDIR)/svm.o: $(SRCDIR)/main.cpp $(SRCDIR)/sequential_solvers.cpp $(DEPS)
	$(CPP) $< $(CPPFLAGS) -c -o $@

$(OUTDIR)/svm: $(OUTDIR)/svm.o
	$(CPP) $< $(CPPFLAGS) -o $@

$(OUTDIR)/cusvm: $(SRCDIR)/main.cpp $(SRCDIR)/cuda_solvers.cu $(DEPS)
	$(CUDACC) $< $(CUDACCFLAGS) -o $@

clean:
	rm -f $(OUTDIR)/svm.o
	rm -f $(OUTDIR)/svm
	rm -f $(OUTDIR)/cusvm
