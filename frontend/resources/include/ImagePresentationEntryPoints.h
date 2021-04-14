/*
 * ImagePresentationEntryPoints.h
 *
 * Copyright (C) 2021 by VISUS (Universitaet Stuttgart).
 * Alle Rechte vorbehalten.
 */

#pragma once

#include <functional>

namespace megamol {
namespace frontend_resources {

struct ImagePresentationEntryPoints {
    std::function<bool(std::string, void*)> add_entry_point;
    std::function<bool(std::string)> remove_entry_point;
    std::function<bool(std::string, std::string)> rename_entry_point;
    std::function<void()> clear_entry_points;
};

} /* end namespace frontend_resources */
} /* end namespace megamol */
