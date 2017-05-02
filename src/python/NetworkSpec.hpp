
#pragma once

#include <iostream>

namespace python {

struct NetworkSpec {
  unsigned numInputs;
  unsigned numOutputs;
  unsigned maxBatchSize;

  inline void Write(std::ostream &out) {
    out << numInputs << std::endl;
    out << numOutputs << std::endl;
    out << maxBatchSize << std::endl;
  }

  static NetworkSpec Read(std::istream &in) {
    NetworkSpec spec;
    in >> spec.numInputs;
    in >> spec.numOutputs;
    in >> spec.maxBatchSize;
    return spec;
  }
};
}
