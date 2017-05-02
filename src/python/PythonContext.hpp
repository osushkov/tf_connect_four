#pragma once

#include <boost/python.hpp>
#include <mutex>

namespace python {

struct PythonContext {
  PythonContext() : threadState(PyThreadState_Get()) { PyEval_ReleaseLock(); }

  ~PythonContext() {
    // PyEval_RestoreThread(threadState);
    Py_Finalize();
  }

  PyThreadState *threadState;
  std::mutex m;
};

struct PythonContextLock {
  PythonContextLock(PythonContext &ctx) : lock(ctx.m) {
    PyEval_AcquireLock();
    PyThreadState_Swap(ctx.threadState);
  }

  ~PythonContextLock() {
    // PyThreadState_Swap(nullptr);
    PyEval_ReleaseLock();
  }

  std::lock_guard<std::mutex> lock;
};
}
