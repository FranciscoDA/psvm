
#include "svm.h"

#ifdef __CUDACC__
  #include "cuda_solvers.cu"
#else
  #include "sequential_solvers.cpp"
#endif

#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>

using namespace std;

struct DatasetError {
  enum ErrorCode {
    INCONSISTENT_D,
    INVALID_Y
  };

  ErrorCode code;
  size_t line;
};

void readCSV(vector<double>& x, vector<double>& y, string path) {
  size_t n = 0;
  size_t d = 0;
  fstream dataset(path, ios_base::in);
  while (!dataset.eof()) {
    string line;
    dataset >> line;
    if (!line.length()) // empty line
      continue;
    stringstream ss(line);
    double val;
    ss >> val;
    ss.ignore(1, ',');
    size_t this_d = 1;
    while (!ss.eof()) {
      x.push_back(val);
      ss >> val;
      ss.ignore(1, ',');
      ++this_d;
    }
    y.push_back(val);
    n++;
    if (val != 1.0 and val != -1.0)
      throw DatasetError {DatasetError::ErrorCode::INVALID_Y, n};
    if (this_d != d) {
      if (n == 1)
        d = this_d;
      else
        throw DatasetError {DatasetError::ErrorCode::INCONSISTENT_D, n};
    }
  }
}

int main(int argc, char** argv)
{
  #ifdef __CUDACC__
    int nCudaDevices;
    cudaGetDeviceCount(&nCudaDevices);
    cout << "Device count: " << nCudaDevices << endl;
    for (int i = 0; i < nCudaDevices; i++) {
      cudaDeviceProp prop;
      cudaGetDeviceProperties(&prop, i);
      cout << "Device #" << i << endl;
      cout << "\tName: " << prop.name << endl;
      cout << "\tMemory Clock (KHz): " << prop.memoryClockRate << endl;
      cout << "\tMemory Bus Width (bits): " << prop.memoryBusWidth << endl;
    }
  #endif
  vector<double> x;
  vector<double> y;

  try {
    readCSV(x, y, "dataset.csv");
  }
  catch (DatasetError e) {
    switch (e.code) {
      case DatasetError::INCONSISTENT_D:
        cerr << "number of attributes in example don't match with the rest of the dataset";
        break;
      case DatasetError::INVALID_Y:
        cerr << "invalid class value";
        break;
      default:
        cerr << "error in dataset";
        break;
    }
    cerr << " at line " << e.line << endl;
    return 1;
  }
  SVM<LinearKernel> svm(10.0, x.size() / y.size(), LinearKernel());
  smo(svm, x, y);
  cout << "No. of SVs: " << svm.getSVAlphaY().size() << endl;
  for(auto ity = begin(svm.getSVAlphaY()); ity != end(svm.getSVAlphaY()); ++ity)
    cout << "alpha_i * y_i = " << *ity << endl;

  mgp(svm, x, y, 0.0001);
  cout << "No. of SVs: " << svm.getSVAlphaY().size() << endl;
  for(auto ity = begin(svm.getSVAlphaY()); ity != end(svm.getSVAlphaY()); ++ity)
    cout << "alpha_i * y_i = " << *ity << endl;
}
