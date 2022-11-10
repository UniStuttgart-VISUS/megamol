/**
 * MegaMol
 * Copyright (c) 2012, MegaMol Dev Team
 * All rights reserved.
 */

#pragma once

#include <filesystem>
#include <string>
#include <vector>

#include "RuntimeConfig.h"

namespace megamol::core::utility {

/**
 * Helper class for generic resource handling.
 *
 * Same as shaders, resources can be placed in configurable directories
 * which can then be automatically searched for the required files.
 */
class ResourceWrapper {
public:
    static std::filesystem::path GetResourcePath(
        frontend_resources::RuntimeConfig const& runtimeConf, const std::string& filename);

    static std::vector<char> LoadResource(
        frontend_resources::RuntimeConfig const& runtimeConf, const std::string& filename);

    static std::string LoadTextResource(
        frontend_resources::RuntimeConfig const& runtimeConf, const std::string& filename);
};

} // namespace megamol::core::utility
