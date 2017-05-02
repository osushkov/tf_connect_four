#pragma once

#include "MatrixView.hpp"
#include <Eigen/Dense>
#include <vector>

using EVector = Eigen::VectorXf;
using EMatrix = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

namespace util {

static inline MatrixView GetMatrixView(EMatrix &m) {
  MatrixView result;
  result.rows = m.rows();
  result.cols = m.cols();
  result.data = m.data();
  return result;
}

double RandInterval(double s, double e);
double GaussianSample(double mean, double sd);

std::vector<float> SoftmaxWeights(const std::vector<float> &in);
float SoftmaxWeightedAverage(const std::vector<float> &in, float temperature);

}
