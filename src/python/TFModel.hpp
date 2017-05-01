#pragma once

#include "util/Common.hpp"
#include <boost/python/numpy.hpp>
#include <vector>

namespace np = boost::python::numpy;

class TFModel {
public:
  TFModel(unsigned batchSize);
  virtual ~TFModel();

  np::ndarray Inference(const np::ndarray &input);
  void SetModelParams(const vector<np::ndarray> &params);

private:
  struct TFModelImpl;
  uptr<TFModelImpl> impl;
};
