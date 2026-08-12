// SPDX-FileCopyrightText: 2026 Stefano Cacciatore
// SPDX-License-Identifier: MIT

#pragma once

#include "kodama/kodama.hpp"

#include <vector>

namespace kodama::detail {

NormalizationResult preprocessing_normalization_cuda(
  const std::vector<float>& train,
  int train_rows,
  const std::vector<float>& test,
  int test_rows,
  int variables,
  const NormalizationOptions& options
);

ScalingResult preprocessing_scaling_cuda(
  const std::vector<float>& train,
  int train_rows,
  const std::vector<float>& test,
  int test_rows,
  int variables,
  const ScalingOptions& options
);

NormalizationResult preprocessing_normalization_metal(
  const std::vector<float>& train,
  int train_rows,
  const std::vector<float>& test,
  int test_rows,
  int variables,
  const NormalizationOptions& options
);

ScalingResult preprocessing_scaling_metal(
  const std::vector<float>& train,
  int train_rows,
  const std::vector<float>& test,
  int test_rows,
  int variables,
  const ScalingOptions& options
);

}  // namespace kodama::detail
