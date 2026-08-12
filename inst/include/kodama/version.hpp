// SPDX-FileCopyrightText: 2026 Stefano Cacciatore
// SPDX-License-Identifier: MIT

#pragma once

#define KODAMA_VERSION_MAJOR 0
#define KODAMA_VERSION_MINOR 1
#define KODAMA_VERSION_PATCH 0
#define KODAMA_VERSION_STRING "0.1.0"

namespace kodama {

inline constexpr int version_major = KODAMA_VERSION_MAJOR;
inline constexpr int version_minor = KODAMA_VERSION_MINOR;
inline constexpr int version_patch = KODAMA_VERSION_PATCH;
inline constexpr const char* version_string = KODAMA_VERSION_STRING;

}  // namespace kodama
