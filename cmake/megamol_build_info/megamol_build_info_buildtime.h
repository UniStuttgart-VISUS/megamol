/**
 * MegaMol
 * Copyright (c) 2021, MegaMol Dev Team
 * All rights reserved.
 */
#pragma once

namespace megamol::build_info {
    extern const uint64_t MEGAMOL_BUILD_TIMESTAMP;
    extern const uint32_t MEGAMOL_GIT_IS_DIRTY;

    extern char const* MEGAMOL_BUILD_TIME;
    extern char const* MEGAMOL_GIT_HASH;
    extern char const* MEGAMOL_GIT_BRANCH_NAME;
    extern char const* MEGAMOL_GIT_BRANCH_NAME_FULL;
    extern char const* MEGAMOL_GIT_ORIGIN_NAME;
    extern char const* MEGAMOL_GIT_REMOTE_URL;

    extern char const* MEGAMOL_GIT_DIFF;
    extern char const* MEGAMOL_CMAKE;
    extern char const* MEGAMOL_LICENSE;
}
