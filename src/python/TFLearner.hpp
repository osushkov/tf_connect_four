#pragma once

#include "../util/Common.hpp"
#include "NetworkSpec.hpp"
#include <boost/python/numpy.hpp>
#include <vector>

namespace np = boost::python::numpy;
namespace bp = boost::python;

namespace python {

class TFLearner {
public:
  TFLearner(const NetworkSpec &spec);
  virtual ~TFLearner();

  // noncopyable
  TFLearner(const TFLearner &other) = delete;
  TFLearner &operator=(TFLearner &other) = delete;

  void LearnIterations(unsigned iters);
  bp::object GetModelParams(void);

private:
  struct TFLearnerImpl;
  uptr<TFLearnerImpl> impl;
};
}
