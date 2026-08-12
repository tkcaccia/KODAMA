// SPDX-FileCopyrightText: 2026 Stefano Cacciatore
// SPDX-License-Identifier: MIT

#include "kodama/kodama.hpp"

#include "preprocessing_backend.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <thread>
#include <vector>

namespace kodama {
namespace {

using Clock = std::chrono::steady_clock;

void validate_matrices(const MatrixView train, const MatrixView test) {
  if (train.data == nullptr || train.rows == 0 || train.cols == 0) {
    throw std::invalid_argument("The training matrix must be non-empty.");
  }
  if (test.data == nullptr) {
    if (test.rows != 0 || test.cols != 0) {
      throw std::invalid_argument("An empty test matrix must have zero dimensions.");
    }
  } else if (test.cols != train.cols) {
    throw std::invalid_argument("Training and test matrices must have the same number of variables.");
  }
}

std::vector<float> copy_float32(const MatrixView matrix) {
  if (matrix.data == nullptr) return {};
  std::vector<float> out(matrix.rows * matrix.cols);
  for (std::size_t row = 0; row < matrix.rows; ++row) {
    for (std::size_t column = 0; column < matrix.cols; ++column) {
      out[row * matrix.cols + column] = matrix.value_float(row, column);
    }
  }
  return out;
}

template <class Function>
void parallel_for(const std::size_t count, const int requested_threads, Function function) {
  const std::size_t workers = std::min<std::size_t>(
    count, static_cast<std::size_t>(std::max(1, requested_threads))
  );
  if (workers <= 1 || count < 2) {
    for (std::size_t i = 0; i < count; ++i) function(i);
    return;
  }
  std::vector<std::thread> threads;
  threads.reserve(workers);
  for (std::size_t worker = 0; worker < workers; ++worker) {
    threads.emplace_back([=, &function]() {
      for (std::size_t i = worker; i < count; i += workers) function(i);
    });
  }
  for (auto& thread : threads) thread.join();
}

float median(std::vector<float>& values, const bool remove_nan) {
  if (remove_nan) {
    values.erase(
      std::remove_if(values.begin(), values.end(), [](const float value) {
        return std::isnan(value);
      }),
      values.end()
    );
  } else {
    for (const float value : values) {
      if (std::isnan(value)) return std::numeric_limits<float>::quiet_NaN();
    }
  }
  if (values.empty()) return std::numeric_limits<float>::quiet_NaN();
  std::sort(values.begin(), values.end());
  const std::size_t middle = values.size() / 2;
  if ((values.size() & 1U) != 0U) return values[middle];
  return (values[middle - 1] + values[middle]) * 0.5f;
}

NormalizationResult normalization_cpu_impl(
  const std::vector<float>& input_train,
  const int train_rows,
  const std::vector<float>& input_test,
  const int test_rows,
  const int variables,
  const NormalizationOptions& options
) {
  const auto started = Clock::now();
  NormalizationResult result;
  result.train = input_train;
  result.test = input_test;
  result.train_rows = static_cast<std::size_t>(train_rows);
  result.test_rows = static_cast<std::size_t>(test_rows);
  result.variables = static_cast<std::size_t>(variables);
  result.method = options.method;
  result.backend = Backend::CPU;
  result.train_coefficients.assign(static_cast<std::size_t>(train_rows), 1.0f);
  result.test_coefficients.assign(static_cast<std::size_t>(test_rows), 1.0f);

  const int threads = std::max(1, options.n_threads);
  if (options.method == NormalizationMethod::PQN) {
    std::vector<float> row_sums(static_cast<std::size_t>(train_rows), 0.0f);
    parallel_for(static_cast<std::size_t>(train_rows), threads, [&](const std::size_t row) {
      float sum = 0.0f;
      for (int column = 0; column < variables; ++column) {
        const float value = result.train[row * variables + column];
        if (!std::isnan(value)) sum += std::abs(value);
      }
      row_sums[row] = sum;
      for (int column = 0; column < variables; ++column) {
        result.train[row * variables + column] /= sum;
      }
    });
    if (!options.reference.empty()) {
      if (options.reference.size() != static_cast<std::size_t>(variables)) {
        throw std::invalid_argument("The PQN reference must contain one value per variable.");
      }
      result.reference = options.reference;
    } else {
      result.reference.resize(static_cast<std::size_t>(variables));
      parallel_for(static_cast<std::size_t>(variables), threads, [&](const std::size_t column) {
        std::vector<float> values(static_cast<std::size_t>(train_rows));
        for (int row = 0; row < train_rows; ++row) {
          values[static_cast<std::size_t>(row)] = result.train[static_cast<std::size_t>(row) * variables + column];
        }
        result.reference[column] = median(values, true);
      });
    }
    parallel_for(static_cast<std::size_t>(train_rows), threads, [&](const std::size_t row) {
      std::vector<float> quotients(static_cast<std::size_t>(variables));
      for (int column = 0; column < variables; ++column) {
        quotients[static_cast<std::size_t>(column)] =
          result.train[row * variables + column] / result.reference[static_cast<std::size_t>(column)];
      }
      const float coefficient = median(quotients, true);
      result.train_coefficients[row] = coefficient * row_sums[row];
      for (int column = 0; column < variables; ++column) {
        result.train[row * variables + column] /= coefficient;
      }
    });
    parallel_for(static_cast<std::size_t>(test_rows), threads, [&](const std::size_t row) {
      float sum = 0.0f;
      for (int column = 0; column < variables; ++column) {
        const float value = result.test[row * variables + column];
        if (!std::isnan(value)) sum += std::abs(value);
      }
      std::vector<float> quotients(static_cast<std::size_t>(variables));
      for (int column = 0; column < variables; ++column) {
        result.test[row * variables + column] /= sum;
        quotients[static_cast<std::size_t>(column)] =
          result.test[row * variables + column] / result.reference[static_cast<std::size_t>(column)];
      }
      const float coefficient = median(quotients, true);
      result.test_coefficients[row] = coefficient * sum;
      for (int column = 0; column < variables; ++column) {
        result.test[row * variables + column] /= coefficient;
      }
    });
  } else if (options.method != NormalizationMethod::None) {
    const auto normalize_rows = [&](std::vector<float>& matrix, std::vector<float>& coefficients,
                                    const int rows, const bool test_matrix) {
      parallel_for(static_cast<std::size_t>(rows), threads, [&](const std::size_t row) {
        float coefficient = 0.0f;
        if (options.method == NormalizationMethod::Median) {
          std::vector<float> values(static_cast<std::size_t>(variables));
          for (int column = 0; column < variables; ++column) {
            values[static_cast<std::size_t>(column)] = matrix[row * variables + column];
          }
          coefficient = median(values, test_matrix);
        } else {
          for (int column = 0; column < variables; ++column) {
            const float value = matrix[row * variables + column];
            if (test_matrix && std::isnan(value)) continue;
            coefficient += options.method == NormalizationMethod::Sqrt ? value * value :
              (test_matrix ? std::abs(value) : value);
          }
          if (options.method == NormalizationMethod::Sqrt) coefficient = std::sqrt(coefficient);
        }
        coefficients[row] = coefficient;
        for (int column = 0; column < variables; ++column) {
          matrix[row * variables + column] /= coefficient;
        }
      });
    };
    normalize_rows(result.train, result.train_coefficients, train_rows, false);
    normalize_rows(result.test, result.test_coefficients, test_rows, true);
  }
  result.runtime_seconds = std::chrono::duration<double>(Clock::now() - started).count();
  return result;
}

ScalingResult scaling_cpu_impl(
  const std::vector<float>& input_train,
  const int train_rows,
  const std::vector<float>& input_test,
  const int test_rows,
  const int variables,
  const ScalingOptions& options
) {
  const auto started = Clock::now();
  ScalingResult result;
  result.train = input_train;
  result.test = input_test;
  result.train_rows = static_cast<std::size_t>(train_rows);
  result.test_rows = static_cast<std::size_t>(test_rows);
  result.variables = static_cast<std::size_t>(variables);
  result.method = options.method;
  result.backend = Backend::CPU;
  result.center.assign(static_cast<std::size_t>(variables), 0.0f);
  result.scale.assign(static_cast<std::size_t>(variables), 1.0f);
  const int threads = std::max(1, options.n_threads);

  if (options.method != ScalingMethod::None) {
    parallel_for(static_cast<std::size_t>(variables), threads, [&](const std::size_t column) {
      float sum = 0.0f;
      for (int row = 0; row < train_rows; ++row) {
        sum += result.train[static_cast<std::size_t>(row) * variables + column];
      }
      const float center = sum / static_cast<float>(train_rows);
      result.center[column] = center;
      if (options.method == ScalingMethod::Autoscaling ||
          options.method == ScalingMethod::ParetoScaling) {
        float squares = 0.0f;
        for (int row = 0; row < train_rows; ++row) {
          const float delta = result.train[static_cast<std::size_t>(row) * variables + column] - center;
          squares += delta * delta;
        }
        const float standard_deviation = std::sqrt(squares / static_cast<float>(train_rows - 1));
        result.scale[column] = options.method == ScalingMethod::ParetoScaling ?
          std::sqrt(standard_deviation) : standard_deviation;
      } else if (options.method == ScalingMethod::RangeScaling) {
        float minimum = result.train[column];
        float maximum = result.train[column];
        for (int row = 1; row < train_rows; ++row) {
          const float value = result.train[static_cast<std::size_t>(row) * variables + column];
          if (std::isnan(value) || std::isnan(minimum) || std::isnan(maximum)) {
            minimum = std::numeric_limits<float>::quiet_NaN();
            maximum = minimum;
            break;
          }
          minimum = std::min(minimum, value);
          maximum = std::max(maximum, value);
        }
        result.scale[column] = maximum - minimum;
      }
    });
    parallel_for(static_cast<std::size_t>(train_rows), threads, [&](const std::size_t row) {
      for (int column = 0; column < variables; ++column) {
        result.train[row * variables + column] =
          (result.train[row * variables + column] - result.center[static_cast<std::size_t>(column)]) /
          result.scale[static_cast<std::size_t>(column)];
      }
    });
    parallel_for(static_cast<std::size_t>(test_rows), threads, [&](const std::size_t row) {
      for (int column = 0; column < variables; ++column) {
        result.test[row * variables + column] =
          (result.test[row * variables + column] - result.center[static_cast<std::size_t>(column)]) /
          result.scale[static_cast<std::size_t>(column)];
      }
    });
  }
  result.runtime_seconds = std::chrono::duration<double>(Clock::now() - started).count();
  return result;
}

}  // namespace

NormalizationResult Normalization_CPU(
  const MatrixView train, const MatrixView test, const NormalizationOptions& options
) {
  validate_matrices(train, test);
  return normalization_cpu_impl(copy_float32(train), static_cast<int>(train.rows),
    copy_float32(test), static_cast<int>(test.rows), static_cast<int>(train.cols), options);
}

NormalizationResult Normalization_CPU(
  const MatrixView train, const NormalizationOptions& options
) { return Normalization_CPU(train, MatrixView{}, options); }

NormalizationResult Normalization_CUDA(
  const MatrixView train, const MatrixView test, const NormalizationOptions& options
) {
  validate_matrices(train, test);
#ifdef KODAMA_ENABLE_CUDA
  return detail::preprocessing_normalization_cuda(copy_float32(train), static_cast<int>(train.rows),
    copy_float32(test), static_cast<int>(test.rows), static_cast<int>(train.cols), options);
#else
  (void)options;
  throw std::runtime_error("The CUDA backend is not available in this build.");
#endif
}

NormalizationResult Normalization_CUDA(
  const MatrixView train, const NormalizationOptions& options
) { return Normalization_CUDA(train, MatrixView{}, options); }

NormalizationResult Normalization_METAL(
  const MatrixView train, const MatrixView test, const NormalizationOptions& options
) {
  validate_matrices(train, test);
#ifdef KODAMA_ENABLE_METAL
  return detail::preprocessing_normalization_metal(copy_float32(train), static_cast<int>(train.rows),
    copy_float32(test), static_cast<int>(test.rows), static_cast<int>(train.cols), options);
#else
  (void)options;
  throw std::runtime_error("The Metal backend is not available in this build.");
#endif
}

NormalizationResult Normalization_METAL(
  const MatrixView train, const NormalizationOptions& options
) { return Normalization_METAL(train, MatrixView{}, options); }

NormalizationResult Normalization(
  const MatrixView train, const MatrixView test, const NormalizationOptions& options
) {
  switch (options.backend) {
    case Backend::CUDA: return Normalization_CUDA(train, test, options);
    case Backend::Metal: return Normalization_METAL(train, test, options);
    case Backend::Auto:
#ifdef KODAMA_ENABLE_CUDA
      return Normalization_CUDA(train, test, options);
#elif defined(KODAMA_ENABLE_METAL)
      return Normalization_METAL(train, test, options);
#else
      return Normalization_CPU(train, test, options);
#endif
    case Backend::CPU: return Normalization_CPU(train, test, options);
  }
  throw std::invalid_argument("Unknown preprocessing backend.");
}

NormalizationResult Normalization(
  const MatrixView train, const NormalizationOptions& options
) { return Normalization(train, MatrixView{}, options); }

ScalingResult Scaling_CPU(
  const MatrixView train, const MatrixView test, const ScalingOptions& options
) {
  validate_matrices(train, test);
  return scaling_cpu_impl(copy_float32(train), static_cast<int>(train.rows),
    copy_float32(test), static_cast<int>(test.rows), static_cast<int>(train.cols), options);
}

ScalingResult Scaling_CPU(
  const MatrixView train, const ScalingOptions& options
) { return Scaling_CPU(train, MatrixView{}, options); }

ScalingResult Scaling_CUDA(
  const MatrixView train, const MatrixView test, const ScalingOptions& options
) {
  validate_matrices(train, test);
#ifdef KODAMA_ENABLE_CUDA
  return detail::preprocessing_scaling_cuda(copy_float32(train), static_cast<int>(train.rows),
    copy_float32(test), static_cast<int>(test.rows), static_cast<int>(train.cols), options);
#else
  (void)options;
  throw std::runtime_error("The CUDA backend is not available in this build.");
#endif
}

ScalingResult Scaling_CUDA(
  const MatrixView train, const ScalingOptions& options
) { return Scaling_CUDA(train, MatrixView{}, options); }

ScalingResult Scaling_METAL(
  const MatrixView train, const MatrixView test, const ScalingOptions& options
) {
  validate_matrices(train, test);
#ifdef KODAMA_ENABLE_METAL
  return detail::preprocessing_scaling_metal(copy_float32(train), static_cast<int>(train.rows),
    copy_float32(test), static_cast<int>(test.rows), static_cast<int>(train.cols), options);
#else
  (void)options;
  throw std::runtime_error("The Metal backend is not available in this build.");
#endif
}

ScalingResult Scaling_METAL(
  const MatrixView train, const ScalingOptions& options
) { return Scaling_METAL(train, MatrixView{}, options); }

ScalingResult Scaling(
  const MatrixView train, const MatrixView test, const ScalingOptions& options
) {
  switch (options.backend) {
    case Backend::CUDA: return Scaling_CUDA(train, test, options);
    case Backend::Metal: return Scaling_METAL(train, test, options);
    case Backend::Auto:
#ifdef KODAMA_ENABLE_CUDA
      return Scaling_CUDA(train, test, options);
#elif defined(KODAMA_ENABLE_METAL)
      return Scaling_METAL(train, test, options);
#else
      return Scaling_CPU(train, test, options);
#endif
    case Backend::CPU: return Scaling_CPU(train, test, options);
  }
  throw std::invalid_argument("Unknown preprocessing backend.");
}

ScalingResult Scaling(
  const MatrixView train, const ScalingOptions& options
) { return Scaling(train, MatrixView{}, options); }

}  // namespace kodama
