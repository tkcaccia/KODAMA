// SPDX-FileCopyrightText: 2026 Stefano Cacciatore
// SPDX-License-Identifier: MIT

#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <map>
#include <random>
#include <string>
#include <vector>

#include "kodama/kodama.hpp"

namespace kodama::detail {

class Timer {
 public:
  Timer() : start_(std::chrono::steady_clock::now()) {}

  double seconds() const {
    const auto now = std::chrono::steady_clock::now();
    return std::chrono::duration<double>(now - start_).count();
  }

 private:
  std::chrono::steady_clock::time_point start_;
};

void validate_inputs(MatrixView x, const std::vector<int>& labels, const std::vector<int>& constrain);

std::vector<int> make_folds(
  const std::vector<int>& labels,
  const std::vector<int>& constrain,
  const FoldOptions& options
);

std::vector<int> unique_labels(const std::vector<int>& labels);

double accuracy(const std::vector<int>& truth, const std::vector<int>& pred);

double accuracy_on_indices(
  const std::vector<int>& truth,
  const std::vector<int>& pred,
  const std::vector<int>& indices
);

ConfusionMatrix make_confusion(
  const std::vector<int>& truth,
  const std::vector<int>& pred
);

double peak_memory_mb();

std::vector<int> indices_where_fold(const std::vector<int>& folds, int fold, bool equal);

std::vector<int> sorted_unique_folds(const std::vector<int>& folds);

}  // namespace kodama::detail
