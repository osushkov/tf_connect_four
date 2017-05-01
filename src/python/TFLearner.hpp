#pragma once

#include "util/Common.hpp"
#include <boost/python/numpy.hpp>
#include <vector>

namespace np = boost::python::numpy;

class TFLearner {
public:
  TFLearner();
  virtual ~TFLearner();

  // noncopyable
  TFLearner(const TFLearner &other) = delete;
  TFLearner &operator=(TFLearner &other) = delete;

  void LearnIterations(unsigned iters);
  vector<np::ndarray> GetModelParams(void);

private:
  struct TFLearnerImpl;
  uptr<TFLearnerImpl> impl;
};
