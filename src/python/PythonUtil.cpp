
#include "PythonUtil.hpp"
#include <boost/python.hpp>
#include <boost/python/numpy.hpp>
#include <boost/python/stl_iterator.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>
#include <mutex>
#include <vector>

using namespace std;

static std::once_flag initialiseFlag;
static bp::object learnerModule;
static bp::object modelModule;

class LearnerInstance {
public:
  LearnerInstance() = default;
  virtual ~LearnerInstance() = default;

  virtual void LearnIterations(unsigned iters) = 0;
  virtual std::vector<np::ndarray> GetModelParams(void) = 0;
};

class PyLearnerInstance final : public LearnerInstance,
                                public bp::wrapper<LearnerInstance> {
public:
  using LearnerInstance::LearnerInstance;

  void LearnIterations(unsigned iters) override {
    get_override("LearnIterations")(iters);
  }

  std::vector<np::ndarray> GetModelParams(void) override {
    return get_override("GetModelParams")();
  }
};

BOOST_PYTHON_MODULE(LearnerFramework) {
  np::initialize();
  bp::class_<PyLearnerInstance, boost::noncopyable>("LearnerInstance");
}

using ArrayList = vector<np::ndarray>;

class ModelInstance {
public:
  virtual ~ModelInstance() = default;

  virtual np::ndarray Inference(const np::ndarray &input) = 0;
  virtual void SetModelParams(const vector<np::ndarray> &params) = 0;
};

class PyModelInstance final : public ModelInstance,
                              public bp::wrapper<ModelInstance> {
public:
  using ModelInstance::ModelInstance;

  np::ndarray Inference(const np::ndarray &input) {
    return get_override("Inference")(input);
  }

  void SetModelParams(const vector<np::ndarray> &params) {
    get_override("SetModelParams")(params);
  }
};

BOOST_PYTHON_MODULE(ModelFramework) {
  np::initialize();

  bp::class_<ArrayList>("ArrayList")
      .def(bp::vector_indexing_suite<ArrayList, true>());
  bp::class_<PyModelInstance, boost::noncopyable>("ModelInstance");
}

void python::Initialise(void) {
  std::call_once(initialiseFlag, []() {
    Py_Initialize();
    PyEval_InitThreads();
    np::initialize();

    PyImport_AppendInittab("LearnerFramework", &initLearnerFramework);
    PyImport_AppendInittab("ModelFramework", &initModelFramework);

    bp::object main = bp::import("__main__");
    bp::object globals = main.attr("__dict__");
    learnerModule =
        python::Import("learner", "src/python/learner.py", globals);
    modelModule = python::Import("model", "src/python/model.py", globals);
  });
}

bp::object &python::GetLearnerModule(void) { return learnerModule; }
bp::object &python::GetModelModule(void) { return modelModule; }

bp::object python::Import(const std::string &module,
                              const std::string &path, bp::object &globals) {
  bp::dict locals;
  locals["module_name"] = module;
  locals["path"] = path;

  bp::exec("import imp\n"
           "import sys\n"
           "sys.path.append('src/python')\n"
           "new_module = imp.load_module(module_name, open(path), path, ('py', "
           "'U', imp.PY_SOURCE))\n",
           globals, locals);
  return locals["new_module"];
}

std::string python::ParseException(void) {
  PyObject *type = nullptr, *value = nullptr, *traceback = nullptr;
  PyErr_Fetch(&type, &value, &traceback);
  std::string ret("Unfetchable Python error");

  if (type != nullptr) {
    bp::handle<> hType(type);
    bp::str typeStr(hType);
    bp::extract<std::string> eTypeStr(typeStr);
    ret = eTypeStr.check() ? eTypeStr() : "Unknown exception type";
  }

  if (value != nullptr) {
    bp::handle<> hVal(value);
    bp::str a(hVal);
    bp::extract<std::string> returned(a);
    ret += returned.check() ? (": " + returned())
                            : std::string(": Unparseable Python error: ");
  }

  if (traceback != nullptr) {
    bp::handle<> hTb(traceback);
    bp::object tb(bp::import("traceback"));
    bp::object fmtTb(tb.attr("format_tb"));
    bp::object tbList(fmtTb(hTb));
    bp::object tbStr(bp::str("\n").join(tbList));
    bp::extract<std::string> returned(tbStr);
    ret += returned.check() ? (": " + returned())
                            : std::string(": Unparseable Python traceback");
  }

  return ret;
}

np::ndarray python::ArrayFromVector(const std::vector<float> &data) {
  bp::tuple shape = bp::make_tuple(data.size());
  bp::tuple stride = bp::make_tuple(sizeof(float));

  return np::from_data(data.data(), np::dtype::get_builtin<float>(), shape,
                       stride, bp::object());
}

std::ostream &operator<<(std::ostream &stream, const np::ndarray &array) {
  stream << bp::extract<char const *>(bp::str(array));
  return stream;
}
