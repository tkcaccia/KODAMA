// SPDX-FileCopyrightText: 2026 Stefano Cacciatore
// SPDX-License-Identifier: MIT
// Independent low-rank spatial covariance screening implementation.

#include "kodama/kodama.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

namespace kodama {
namespace {

using Clock = std::chrono::steady_clock;
constexpr float kScales[] = {0.1f, 0.2f, 0.4f, 0.8f, 1.6f};

struct SampleGroup {
  int label = 0;
  std::vector<int> rows;
};

template <class Function>
void parallel_for(const std::size_t count, const int requested,
                  Function function) {
  const std::size_t workers =
      std::min<std::size_t>(count, std::max(1, requested));
  if (workers <= 1) {
    for (std::size_t i = 0; i < count; ++i)
      function(i);
    return;
  }
  std::vector<std::thread> threads;
  for (std::size_t worker = 0; worker < workers; ++worker) {
    threads.emplace_back([=, &function]() {
      for (std::size_t i = worker; i < count; i += workers)
        function(i);
    });
  }
  for (auto &thread : threads)
    thread.join();
}

std::vector<SampleGroup> sample_groups(const std::size_t rows,
                                       const std::vector<int> &samples) {
  if (samples.empty()) {
    SampleGroup group;
    group.rows.resize(rows);
    std::iota(group.rows.begin(), group.rows.end(), 0);
    return {std::move(group)};
  }
  std::vector<SampleGroup> groups;
  std::unordered_map<int, std::size_t> locations;
  for (std::size_t row = 0; row < samples.size(); ++row) {
    const auto inserted = locations.emplace(samples[row], groups.size());
    if (inserted.second)
      groups.push_back(SampleGroup{samples[row], {}});
    groups[inserted.first->second].rows.push_back(static_cast<int>(row));
  }
  return groups;
}

void validate(const MatrixView data, const MatrixView spatial,
              const std::vector<int> &samples) {
  if (data.data == nullptr || data.rows < 8 || data.cols == 0) {
    throw std::invalid_argument(
        "data must contain at least eight rows and one variable.");
  }
  if (spatial.data == nullptr || spatial.rows != data.rows ||
      (spatial.cols != 2 && spatial.cols != 3)) {
    throw std::invalid_argument(
        "spatial must have the data rows and two or three columns.");
  }
  if (!samples.empty() && samples.size() != data.rows) {
    throw std::invalid_argument("samples must contain one value per data row.");
  }
  for (std::size_t row = 0; row < data.rows; ++row) {
    for (std::size_t column = 0; column < data.cols; ++column) {
      if (!std::isfinite(data.value_float(row, column)))
        throw std::invalid_argument("data values must be finite.");
    }
    for (std::size_t column = 0; column < spatial.cols; ++column) {
      if (!std::isfinite(spatial.value_float(row, column)))
        throw std::invalid_argument("spatial coordinates must be finite.");
    }
  }
}

void append_orthonormal_basis(const std::vector<float> &columns, const int rows,
                              const int column_count, std::vector<float> &bases,
                              std::vector<int> &offsets) {
  if (rows <= 0 || column_count <= 0)
    return;
  std::vector<float> accepted;
  for (int column = 0; column < column_count; ++column) {
    std::vector<float> vector(static_cast<std::size_t>(rows));
    float mean = 0.0f;
    for (int row = 0; row < rows; ++row)
      mean += columns[static_cast<std::size_t>(column) * rows + row];
    mean /= static_cast<float>(rows);
    for (int row = 0; row < rows; ++row)
      vector[static_cast<std::size_t>(row)] =
          columns[static_cast<std::size_t>(column) * rows + row] - mean;
    for (std::size_t previous = 0; previous < accepted.size() / rows;
         ++previous) {
      float dot = 0.0f;
      for (int row = 0; row < rows; ++row)
        dot += vector[static_cast<std::size_t>(row)] *
               accepted[previous * rows + static_cast<std::size_t>(row)];
      for (int row = 0; row < rows; ++row)
        vector[static_cast<std::size_t>(row)] -=
            dot * accepted[previous * rows + static_cast<std::size_t>(row)];
    }
    float norm = 0.0f;
    for (const float value : vector)
      norm += value * value;
    norm = std::sqrt(norm);
    if (!(norm > 1.0e-6f))
      continue;
    for (float &value : vector)
      value /= norm;
    accepted.insert(accepted.end(), vector.begin(), vector.end());
  }
  bases.insert(bases.end(), accepted.begin(), accepted.end());
  offsets.push_back(
      static_cast<int>(bases.size() / static_cast<std::size_t>(rows)));
}

void build_bases(const std::vector<float> &coordinates, const int rows,
                 const int dimensions, std::vector<float> &bases,
                 std::vector<int> &offsets) {
  std::vector<float> normalized(static_cast<std::size_t>(dimensions) * rows);
  for (int dimension = 0; dimension < dimensions; ++dimension) {
    float low = coordinates[static_cast<std::size_t>(dimension)];
    float high = low;
    for (int row = 1; row < rows; ++row) {
      const float value =
          coordinates[static_cast<std::size_t>(row) * dimensions + dimension];
      low = std::min(low, value);
      high = std::max(high, value);
    }
    const float span = high - low;
    for (int row = 0; row < rows; ++row)
      normalized[static_cast<std::size_t>(dimension) * rows + row] =
          span > 0.0f
              ? (coordinates[static_cast<std::size_t>(row) * dimensions +
                             dimension] -
                 low) /
                    span
              : 0.0f;
  }
  offsets.clear();
  offsets.push_back(0);
  append_orthonormal_basis(normalized, rows, dimensions, bases, offsets);
  for (const float scale : kScales) {
    std::vector<float> transformed(static_cast<std::size_t>(dimensions) * rows);
    for (int dimension = 0; dimension < dimensions; ++dimension)
      for (int row = 0; row < rows; ++row) {
        const float value =
            normalized[static_cast<std::size_t>(dimension) * rows + row];
        transformed[static_cast<std::size_t>(dimension) * rows + row] =
            std::exp(-(value * value) / (2.0f * scale * scale));
      }
    append_orthonormal_basis(transformed, rows, dimensions, bases, offsets);
  }
  constexpr float pi = 3.14159265358979323846f;
  for (const float period : kScales) {
    std::vector<float> transformed(static_cast<std::size_t>(dimensions * 2) *
                                   rows);
    for (int dimension = 0; dimension < dimensions; ++dimension)
      for (int row = 0; row < rows; ++row) {
        const float angle =
            2.0f * pi *
            normalized[static_cast<std::size_t>(dimension) * rows + row] /
            period;
        transformed[static_cast<std::size_t>(2 * dimension) * rows + row] =
            std::sin(angle);
        transformed[static_cast<std::size_t>(2 * dimension + 1) * rows + row] =
            std::cos(angle);
      }
    append_orthonormal_basis(transformed, rows, dimensions * 2, bases, offsets);
  }
}

std::vector<float> cpu_statistics(const std::vector<float> &values,
                                  const int rows, const int variables,
                                  const std::vector<float> &bases,
                                  const std::vector<int> &offsets,
                                  const int threads) {
  const int tests = static_cast<int>(offsets.size()) - 1;
  std::vector<float> output(static_cast<std::size_t>(tests) * variables, 0.0f);
  parallel_for(
      static_cast<std::size_t>(variables), threads,
      [&](const std::size_t variable) {
        const float *x = values.data() + variable * rows;
        float mean = 0.0f;
        for (int row = 0; row < rows; ++row)
          mean += x[row];
        mean /= rows;
        float denominator = 0.0f;
        for (int row = 0; row < rows; ++row) {
          const float v = x[row] - mean;
          denominator += v * v;
        }
        if (!(denominator > std::numeric_limits<float>::epsilon()))
          return;
        for (int test = 0; test < tests; ++test) {
          float explained = 0.0f;
          for (int basis = offsets[test]; basis < offsets[test + 1]; ++basis) {
            float dot = 0.0f;
            const float *q =
                bases.data() + static_cast<std::size_t>(basis) * rows;
            for (int row = 0; row < rows; ++row)
              dot += q[row] * (x[row] - mean);
            explained += dot * dot;
          }
          output[static_cast<std::size_t>(test) * variables + variable] =
              std::min(1.0f, std::max(0.0f, explained / denominator));
        }
      });
  return output;
}

double beta_fraction(const double a, const double b, const double x) {
  constexpr int maximum = 200;
  constexpr double epsilon = 3.0e-14;
  constexpr double floor = 1.0e-300;
  const double qab = a + b, qap = a + 1.0, qam = a - 1.0;
  double c = 1.0, d = 1.0 - qab * x / qap;
  if (std::abs(d) < floor)
    d = floor;
  d = 1.0 / d;
  double h = d;
  for (int m = 1; m <= maximum; ++m) {
    const int m2 = 2 * m;
    double aa = m * (b - m) * x / ((qam + m2) * (a + m2));
    d = 1.0 + aa * d;
    if (std::abs(d) < floor)
      d = floor;
    c = 1.0 + aa / c;
    if (std::abs(c) < floor)
      c = floor;
    d = 1.0 / d;
    h *= d * c;
    aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));
    d = 1.0 + aa * d;
    if (std::abs(d) < floor)
      d = floor;
    c = 1.0 + aa / c;
    if (std::abs(c) < floor)
      c = floor;
    d = 1.0 / d;
    const double delta = d * c;
    h *= delta;
    if (std::abs(delta - 1.0) < epsilon)
      break;
  }
  return h;
}

double regularized_beta(const double x, const double a, const double b) {
  if (x <= 0.0)
    return 0.0;
  if (x >= 1.0)
    return 1.0;
  const double front =
      std::exp(std::lgamma(a + b) - std::lgamma(a) - std::lgamma(b) +
               a * std::log(x) + b * std::log1p(-x));
  if (x < (a + 1.0) / (a + b + 2.0))
    return front * beta_fraction(a, b, x) / a;
  return 1.0 - front * beta_fraction(b, a, 1.0 - x) / b;
}

double projection_p(const float r2, const int dimensions, const int rows) {
  if (dimensions <= 0 || rows <= dimensions + 1)
    return 1.0;
  return std::max(1.0e-300,
                  std::min(1.0, regularized_beta(1.0 - static_cast<double>(r2),
                                                 0.5 * (rows - dimensions - 1),
                                                 0.5 * dimensions)));
}

double cauchy_combine(const std::vector<double> &p_values) {
  if (p_values.empty())
    return 1.0;
  long double sum = 0.0L;
  for (const double value : p_values) {
    const long double p =
        std::max(1e-15L, std::min(1.0L - 1e-15L, (long double)value));
    sum += std::tan((0.5L - p) * 3.14159265358979323846L);
  }
  return std::max(
      1.0e-300, std::min(1.0, (double)(0.5L - std::atan(sum / p_values.size()) /
                                                  3.14159265358979323846L)));
}

std::vector<double> adjust_bh(const std::vector<double> &p_values) {
  std::vector<int> order(p_values.size());
  std::iota(order.begin(), order.end(), 0);
  std::sort(order.begin(), order.end(),
            [&](int a, int b) { return p_values[a] < p_values[b]; });
  std::vector<double> adjusted(p_values.size(), 1.0);
  double running = 1.0;
  for (std::size_t reverse = order.size(); reverse > 0; --reverse) {
    const int index = order[reverse - 1];
    running = std::min(running, p_values[index] * order.size() / reverse);
    adjusted[index] = std::min(1.0, running);
  }
  return adjusted;
}

SpatialFeatureResult run(const MatrixView data, const MatrixView spatial,
                         const std::vector<int> &samples,
                         const SpatialFeatureOptions &options) {
  validate(data, spatial, samples);
  const auto started = Clock::now();
  const auto groups = sample_groups(data.rows, samples);
  for (const auto &group : groups)
    if (group.rows.size() < 8)
      throw std::invalid_argument(
          "Every slide must contain at least eight rows.");
  SpatialFeatureResult result;
  result.samples = data.rows;
  result.variables = data.cols;
  result.sample_groups = groups.size();
  result.backend = Backend::CPU;
  result.per_sample_score.assign(groups.size() * data.cols, 0.0f);
  result.per_sample_p_value.assign(groups.size() * data.cols, 1.0);
  std::vector<unsigned char> zero_in_sample(data.cols, 0);
  for (std::size_t group_index = 0; group_index < groups.size();
       ++group_index) {
    const auto &group = groups[group_index];
    result.sample_labels.push_back(group.label);
    const int rows = static_cast<int>(group.rows.size());
    std::vector<float> coordinates(group.rows.size() * spatial.cols),
        feature_major(data.cols * group.rows.size());
    for (std::size_t local = 0; local < group.rows.size(); ++local) {
      const std::size_t global = group.rows[local];
      for (std::size_t d = 0; d < spatial.cols; ++d)
        coordinates[local * spatial.cols + d] = spatial.value_float(global, d);
      for (std::size_t v = 0; v < data.cols; ++v)
        feature_major[v * group.rows.size() + local] =
            data.value_float(global, v);
    }
    for (std::size_t v = 0; v < data.cols; ++v) {
      bool all_zero = true;
      for (std::size_t row = 0; row < group.rows.size(); ++row)
        if (feature_major[v * group.rows.size() + row] != 0.0f) {
          all_zero = false;
          break;
        }
      if (all_zero)
        zero_in_sample[v] = 1;
    }
    auto phase = Clock::now();
    std::vector<float> bases;
    std::vector<int> offsets;
    build_bases(coordinates, rows, static_cast<int>(spatial.cols), bases,
                offsets);
    result.basis_seconds +=
        std::chrono::duration<double>(Clock::now() - phase).count();
    if (result.basis_dimensions.empty())
      for (std::size_t t = 0; t + 1 < offsets.size(); ++t)
        result.basis_dimensions.push_back(offsets[t + 1] - offsets[t]);
    phase = Clock::now();
    const std::vector<float> statistics = cpu_statistics(
      feature_major, rows, static_cast<int>(data.cols), bases, offsets,
      options.n_threads
    );
    result.statistic_seconds +=
        std::chrono::duration<double>(Clock::now() - phase).count();
    const int tests = static_cast<int>(offsets.size()) - 1;
    for (std::size_t v = 0; v < data.cols; ++v) {
      std::vector<double> test_p;
      for (int t = 0; t < tests; ++t)
        test_p.push_back(projection_p(
            statistics[static_cast<std::size_t>(t) * data.cols + v],
            offsets[t + 1] - offsets[t], rows));
      const double p = cauchy_combine(test_p);
      result.per_sample_p_value[group_index * data.cols + v] = p;
      result.per_sample_score[group_index * data.cols + v] =
          static_cast<float>(-std::log10(p));
    }
  }
  result.p_value.assign(data.cols, 1.0);
  result.score.assign(data.cols, 0.0f);
  for (std::size_t v = 0; v < data.cols; ++v) {
    if (options.require_nonzero_each_sample && zero_in_sample[v])
      continue;
    std::vector<double> slide_p;
    for (std::size_t g = 0; g < groups.size(); ++g) {
      const double p = result.per_sample_p_value[g * data.cols + v];
      slide_p.push_back(p);
      result.score[v] =
          std::max(result.score[v], static_cast<float>(-std::log10(p)));
    }
    result.p_value[v] = cauchy_combine(slide_p);
  }
  result.adjusted_p_value = adjust_bh(result.p_value);
  result.ranking.resize(data.cols);
  std::iota(result.ranking.begin(), result.ranking.end(), 0);
  std::stable_sort(
      result.ranking.begin(), result.ranking.end(),
      [&](int a, int b) { return result.score[a] > result.score[b]; });
  result.runtime_seconds =
      std::chrono::duration<double>(Clock::now() - started).count();
  return result;
}

} // namespace

SpatialFeatureResult
SpatialFeatureSelection_CPU(const MatrixView d, const MatrixView s,
                            const std::vector<int> &g,
                            const SpatialFeatureOptions &o) {
  return run(d, s, g, o);
}
SpatialFeatureResult SpatialFeatureSelection(const MatrixView d,
                                             const MatrixView s,
                                             const std::vector<int> &g,
                                             const SpatialFeatureOptions &o) {
  return SpatialFeatureSelection_CPU(d, s, g, o);
}

} // namespace kodama
