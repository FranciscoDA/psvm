#pragma once

#include <iostream>
#include "kernels.h"
#include "classifiers.h"

void toStream(const Kernel* kernel, std::ostream& stream);
void toStream(const CSVC* svc, std::ostream& stream);

Kernel* kernelFromStream(std::istream& stream);
CSVC* csvcFromStream(std::istream& stream);
