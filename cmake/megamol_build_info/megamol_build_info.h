/**
 * MegaMol
 * Copyright (c) 2021, MegaMol Dev Team
 * All rights reserved.
 */

#include <string>

namespace megamol::build_info {
    // Configure time
    extern const int MEGAMOL_VERSION_MAJOR;
    extern const int MEGAMOL_VERSION_MINOR;

    // Build time
    extern const std::string MEGAMOL_GIT_HASH;
    extern const std::string MEGAMOL_GIT_BRANCH_NAME;
    extern const std::string MEGAMOL_GIT_BRANCH_NAME_FULL;
    extern const std::string MEGAMOL_GIT_ORIGIN_NAME;
    extern const std::string MEGAMOL_GIT_REMOTE_URL;
    extern const std::string MEGAMOL_GIT_DIFF;
    extern const std::string MEGAMOL_CMAKE_CACHE;
}
