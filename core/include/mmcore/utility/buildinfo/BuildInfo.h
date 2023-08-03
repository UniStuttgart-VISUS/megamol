/**
 * MegaMol
 * Copyright (c) 2021, MegaMol Dev Team
 * All rights reserved.
 */
#pragma once

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace megamol::core::utility::buildinfo {
// Configure time
int MEGAMOL_VERSION_MAJOR();
int MEGAMOL_VERSION_MINOR();
int MEGAMOL_VERSION_PATCH();
std::string const& MEGAMOL_VERSION();

// Build time
std::string const& MEGAMOL_GIT_HASH();
std::string const& MEGAMOL_GIT_BRANCH_NAME();
std::string const& MEGAMOL_GIT_BRANCH_NAME_FULL();
std::string const& MEGAMOL_GIT_REMOTE_NAME();
std::string const& MEGAMOL_GIT_REMOTE_URL();
std::string const& MEGAMOL_GIT_DIFF();
int MEGAMOL_GIT_IS_DIRTY();
std::string const& MEGAMOL_GIT_LAST_COMMIT_DATE();
std::string const& MEGAMOL_BUILD_CONFIG();
std::string const& MEGAMOL_LICENSE();
std::string const& MEGAMOL_CMAKE_CACHE();

// String list of all build info values
std::vector<std::pair<std::string, std::string>> const& AllKeyValues();
} // namespace megamol::core::utility::buildinfo
