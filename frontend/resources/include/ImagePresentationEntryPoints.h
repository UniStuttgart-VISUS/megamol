/*
 * ImagePresentationEntryPoints.h
 *
 * Copyright (C) 2021 by VISUS (Universitaet Stuttgart).
 * Alle Rechte vorbehalten.
 */

#pragma once

#include "ImageWrapper.h"
#include "FrontendResource.h"

#include <functional>

namespace megamol {
namespace frontend_resources {

using EntryPointExecutionCallback =
    std::function<bool(
          void*
        , std::vector<megamol::frontend::FrontendResource> const&
        , ImageWrapper&
        )>;

using EntryPointRenderFunctions =
    std::tuple<
        // ptr to entry point object
        void*,
        // rendering execution function
        EntryPointExecutionCallback,
        // get requested resources function
        std::function<std::vector<std::string>()>
    >;

struct ImagePresentationEntryPoints {
    std::function<bool(std::string, EntryPointRenderFunctions const&)> add_entry_point;
    std::function<bool(std::string)> remove_entry_point;
    std::function<bool(std::string, std::string)> rename_entry_point;
    std::function<void()> clear_entry_points;
};

} /* end namespace frontend_resources */
} /* end namespace megamol */
