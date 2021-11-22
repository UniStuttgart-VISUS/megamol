/**
 * MegaMol
 * Copyright (c) 2021, MegaMol Dev Team
 * All rights reserved.
 */
#pragma once

#include <cstdint>

namespace megamol::build_info {
// Configure time
extern int const MEGAMOL_VERSION_MAJOR;
extern int const MEGAMOL_VERSION_MINOR;
extern int const MEGAMOL_VERSION_PATCH;
extern char const* const MEGAMOL_VERSION;

// Build time
extern uint64_t const MEGAMOL_BUILD_TIMESTAMP;
extern char const* const MEGAMOL_BUILD_TIME;
extern char const* const MEGAMOL_GIT_HASH;
extern char const* const MEGAMOL_GIT_BRANCH_NAME;
extern char const* const MEGAMOL_GIT_BRANCH_NAME_FULL;
extern char const* const MEGAMOL_GIT_REMOTE_NAME;
extern char const* const MEGAMOL_GIT_REMOTE_URL;
extern char const* const MEGAMOL_GIT_DIFF;
extern int const MEGAMOL_GIT_IS_DIRTY;
extern char const* const MEGAMOL_LICENSE;
extern char const* const MEGAMOL_CMAKE_CACHE;
} // namespace megamol::build_info
