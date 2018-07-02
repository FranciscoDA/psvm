
CPP=g++
CUDACC=nvcc

CPPFLAGS=-std=c++14

CUDAARCH=--gpu-architecture compute_35
CUDACCFLAGS=-std=c++14 --expt-extended-lambda -x cu $(CUDAARCH)

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
$(CPPOUTDIR)/svm: $(CPPOUTDIR)/main.o $(CPPOUTDIR)/io_formats.o $(CPPDEPS)
	$(CPP) $(filter-out $(CPPDEPS),$^) $(CPPFLAGS) -o $@

# nvcc cant link c++ code -> compile everything in one go
$(CUDAOUTDIR)/%.o: $(SRCDIR)/%.cpp $(DEPS)
	$(CUDACC) $< $(CUDACCFLAGS) -dc -o $@
$(CUDAOUTDIR)/cusvm: $(CUDAOUTDIR)/io_formats.o $(CUDAOUTDIR)/main.o $(CUDADEPS)
	$(CUDACC) -dlink $(CUDAARCH) $(filter-out $(CUDADEPS),$^) -o $(CUDAOUTDIR)/device_code.o
	$(CUDACC) -link $(CUDAARCH) $(CUDAOUTDIR)/device_code.o $(filter-out $(CUDADEPS),$^) -o $@

clean:
	rm -rf $(CPPOUTDIR)
	rm -rf $(CUDAOUTDIR)
