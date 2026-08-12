// SPDX-FileCopyrightText: 2026 Stefano Cacciatore
// SPDX-License-Identifier: MIT

#pragma once

#include <vector>

namespace kodama::detail {

std::vector<float> cuda_pca_matrix_multiply(
  const std::vector<float>& left,
  int left_rows,
  int left_cols,
  const std::vector<float>& right,
  int right_rows,
  int right_cols,
  bool transpose_left,
  bool transpose_right,
  int gpu_device
);

}  // namespace kodama::detail
