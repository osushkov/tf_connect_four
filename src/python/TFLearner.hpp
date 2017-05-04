#pragma once

#include "../util/Common.hpp"
#include "NetworkSpec.hpp"
#include "PythonContext.hpp"
#include <boost/python/numpy.hpp>
#include <vector>
#include <cassert>

namespace np = boost::python::numpy;
namespace bp = boost::python;

namespace python {

struct QLearnBatch {
  QLearnBatch()
      : initialStates(
            np::empty(bp::make_tuple(1, 1), np::dtype::get_builtin<float>())),
        successorStates(
            np::empty(bp::make_tuple(1, 1), np::dtype::get_builtin<float>())),
        actionsTaken(
            np::empty(bp::make_tuple(1), np::dtype::get_builtin<int>())),
        isEndStateTerminal(
            np::empty(bp::make_tuple(1), np::dtype::get_builtin<bool>())),
        rewardsGained(
            np::empty(bp::make_tuple(1), np::dtype::get_builtin<float>())),
        futureRewardDiscount(1.0f), learnRate(0.0f) {}

  QLearnBatch(const np::ndarray &initialStates,
              const np::ndarray &successorStates,
              const np::ndarray &actionsTaken,
              const np::ndarray &isEndStateTerminal,
              const np::ndarray &rewardsGained, float futureRewardDiscount,
              float learnRate)
      : initialStates(initialStates), successorStates(successorStates),
        actionsTaken(actionsTaken), isEndStateTerminal(isEndStateTerminal),
        rewardsGained(rewardsGained),
        futureRewardDiscount(futureRewardDiscount), learnRate(learnRate) {
    assert(learnRate > 0.0f);
  }

  np::ndarray initialStates;
  np::ndarray successorStates;

  // 1D arrays.
  np::ndarray actionsTaken; // action indices
  np::ndarray isEndStateTerminal; // boolean
  np::ndarray rewardsGained; // floats

  float futureRewardDiscount;
  float learnRate;
};

class TFLearner {
public:
  TFLearner(PythonThreadContext &ctx, const NetworkSpec &spec);
  virtual ~TFLearner();

  // noncopyable
  TFLearner(const TFLearner &other) = delete;
  TFLearner &operator=(TFLearner &other) = delete;

  void Learn(PythonThreadContext &ctx, const QLearnBatch &batch);
  void UpdateTargetParams(PythonThreadContext &ctx);
  np::ndarray QFunction(PythonThreadContext &ctx, const np::ndarray &state);

private:
  struct TFLearnerImpl;
  uptr<TFLearnerImpl> impl;
};
}
