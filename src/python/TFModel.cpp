#include "TFModel.hpp"
#include "PythonUtil.hpp"

#include <boost/python.hpp>
#include <boost/python/numeric.hpp>
#include <boost/python/numpy.hpp>
#include <boost/python/stl_iterator.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>
#include <cassert>
#include <iostream>
#include <mutex>
#include <vector>

namespace np = boost::python::numpy;
namespace bp = boost::python;

struct TFModel::TFModelImpl {
  bp::object model;

  TFModelImpl(unsigned batchSize) {
    assert(batchSize >= 1);

    try {
      bp::object Model = PythonUtil::GetModelModule().attr("Model");
      model = Model(batchSize);
    } catch (const bp::error_already_set &e) {
      std::cerr << std::endl << PythonUtil::ParseException() << std::endl;
      throw e;
    }
  }

  np::ndarray Inference(const np::ndarray &input) {
    try {
      return bp::extract<np::ndarray>(model.attr("Inference")(input));
    } catch (const bp::error_already_set &e) {
      std::cerr << std::endl << PythonUtil::ParseException() << std::endl;
      throw e;
    }
  }

  void SetModelParams(const vector<np::ndarray> &params) {
    try {
      model.attr("SetModelParams")(params);
    } catch (const bp::error_already_set &e) {
      std::cerr << std::endl << PythonUtil::ParseException() << std::endl;
      throw e;
    }
  }
};

TFModel::TFModel(unsigned batchSize) : impl(new TFModelImpl(batchSize)) {}
TFModel::~TFModel() = default;

np::ndarray TFModel::Inference(const np::ndarray &input) {
  return impl->Inference(input);
}

void TFModel::SetModelParams(const vector<np::ndarray> &params) {
  impl->SetModelParams(params);
}
