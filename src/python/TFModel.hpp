#pragma once

#include "../util/Common.hpp"
#include "NetworkSpec.hpp"
#include <boost/python/numpy.hpp>

namespace np = boost::python::numpy;
namespace bp = boost::python;

namespace python {

class TFModel {
public:
  TFModel(const NetworkSpec &spec);
  virtual ~TFModel();

  // noncopyable
  TFModel(const TFModel &other) = delete;
  TFModel &operator=(TFModel &other) = delete;

  np::ndarray Inference(const np::ndarray &input);
  void SetModelParams(const bp::object &params);

private:
  struct TFModelImpl;
  uptr<TFModelImpl> impl;
};
}
