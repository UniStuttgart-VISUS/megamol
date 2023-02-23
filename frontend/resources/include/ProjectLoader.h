/*
 * ProjectLoader.h
 *
 * Copyright (C) 2020 by VISUS (Universitaet Stuttgart).
 * Alle Rechte vorbehalten.
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
