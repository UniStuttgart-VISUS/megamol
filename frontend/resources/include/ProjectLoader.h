/**
 * MegaMol
 * Copyright (c) 2020, MegaMol Dev Team
 * All rights reserved.
 */

#pragma once

#include <filesystem>
#include <functional>
#include <string>

namespace megamol::frontend_resources {

struct ProjectLoader {
    std::function<bool(std::filesystem::path const& /*filename*/)> load_filename; // returns false if loading failed
};

} // namespace megamol::frontend_resources
