/*
 * ProjectLoader.h
 *
 * Copyright (C) 2020 by VISUS (Universitaet Stuttgart).
 * Alle Rechte vorbehalten.
 */

#pragma once

#include <functional>
#include <string>

namespace megamol {
namespace frontend_resources {

struct ProjectLoader {
    std::function<bool(std::string const& /*filename*/)> load_filename; // returns false if loading failed
};

} /* end namespace frontend_resources */
} /* end namespace megamol */
