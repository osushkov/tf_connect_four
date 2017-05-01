#include "TFLearner.hpp"
#include "PythonUtil.hpp"

#include <boost/python.hpp>
#include <boost/python/numpy.hpp>
#include <boost/python/stl_iterator.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>
#include <iostream>
#include <mutex>
#include <vector>

namespace np = boost::python::numpy;
namespace bp = boost::python;

struct TFLearner::TFLearnerImpl {
  bp::object learner;

  TFLearnerImpl() {
    try {
      bp::object Learner = PythonUtil::GetLearnerModule().attr("Learner");
      learner = Learner();
    } catch (const bp::error_already_set &e) {
      std::cerr << std::endl << PythonUtil::ParseException() << std::endl;
      throw e;
    }
  }

  void LearnIterations(unsigned iters) {
    try {
      learner.attr("LearnIterations")(iters);
    } catch (const bp::error_already_set &e) {
      std::cerr << std::endl << PythonUtil::ParseException() << std::endl;
      throw e;
    }
  }

  std::vector<np::ndarray> GetModelParams(void) {
    try {
      return PythonUtil::ToStdVector<np::ndarray>(
          learner.attr("GetModelParams")());
    } catch (const bp::error_already_set &e) {
      std::cerr << std::endl << PythonUtil::ParseException() << std::endl;
      throw e;
    }
  }
};

TFLearner::TFLearner() : impl(new TFLearnerImpl()) {}
TFLearner::~TFLearner() = default;

void TFLearner::LearnIterations(unsigned iters) {
  impl->LearnIterations(iters);
}

std::vector<np::ndarray> TFLearner::GetModelParams(void) {
  return impl->GetModelParams();
}
