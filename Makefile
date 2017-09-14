
CPP=g++
CUDACC=nvcc
CPPFLAGS=-std=c++11
CUDACCFLAGS=-std c++11 --expt-extended-lambda -x cu --gpu-architecture compute_35 -lcuda -lcudart
SRCDIR=src
DEPS=$(SRCDIR)/svm.h $(SRCDIR)/io_formats.h $(SRCDIR)/classifier.h

CPPOUTDIR=bin
CPPDEPS=$(SRCDIR)/sequential_solvers.h $(DEPS)

CUDAOUTDIR=cubin
CUDADEPS=$(SRCDIR)/cuda_solvers.cu $(DEPS)

all: $(CPPOUTDIR) $(CUDAOUTDIR) $(CPPOUTDIR)/svm $(CUDAOUTDIR)/cusvm

$(CPPOUTDIR):
	mkdir -p $@
$(CUDAOUTDIR):
	mkdir -p $@

# sane, modular, C++ compliant compilation+linkage workflow
$(CPPOUTDIR)/%.o: $(SRCDIR)/%.cpp $(DEPS)
	$(CPP) $< $(CPPFLAGS) -o $@ -c
$(CPPOUTDIR)/svm: $(SRCDIR)/main.cpp $(CPPOUTDIR)/io_formats.o $(CPPDEPS)
	$(CPP) $(filter-out $(CPPDEPS),$^) $(CPPFLAGS) -o $@

# nvcc cant link c++ code -> compile everything in one go
$(CUDAOUTDIR)/cusvm: $(SRCDIR)/main.cpp $(SRCDIR)/io_formats.cpp $(CUDADEPS)
	$(CUDACC) $(filter-out $(CUDADEPS),$^) $(CUDACCFLAGS) -o $@

clean:
	rm -rf $(CPPOUTDIR)
	rm -rf $(CUDAOUTDIR)
