#pragma once

#include "../util/Common.hpp"
#include "NetworkSpec.hpp"
#include <boost/python/numpy.hpp>
#include <vector>

namespace np = boost::python::numpy;
namespace bp = boost::python;

namespace python {

struct QLearnBatch {
  QLearnBatch()
      : initialStates(
            np::empty(bp::make_tuple(1, 1), np::dtype::get_builtin<float>())),
        successorStates(
            np::empty(bp::make_tuple(1, 1), np::dtype::get_builtin<float>())),
        futureRewardDiscount(1.0f) {}

  QLearnBatch(const np::ndarray &initialStates,
              const np::ndarray &successorStates)
      : initialStates(initialStates), successorStates(successorStates),
        futureRewardDiscount(1.0f) {}

  np::ndarray initialStates;
  np::ndarray successorStates;

  // 1D arrays.
  // np::ndarray actionsTaken; // action indices
  // np::ndarray isEndStateTerminal; // boolean
  // np::ndarray rewardsGained; // floats

  float futureRewardDiscount;
};

class TFLearner {
public:
  TFLearner(const NetworkSpec &spec);
  virtual ~TFLearner();

  // noncopyable
  TFLearner(const TFLearner &other) = delete;
  TFLearner &operator=(TFLearner &other) = delete;

  void Learn(const QLearnBatch &batch);
  void UpdateTargetParams(void);
  np::ndarray QFunction(const np::ndarray &state);

private:
  struct TFLearnerImpl;
  uptr<TFLearnerImpl> impl;
};
}
