/*
 * EntryPoint.h
 *
 * Copyright (C) 2021 by VISUS (Universitaet Stuttgart).
 * Alle Rechte vorbehalten.
 */

#pragma once

#include "ImageWrapper.h"
#include "FrontendResource.h"

namespace megamol {
namespace frontend_resources {

struct RenderInputsUpdate {
    virtual ~RenderInputsUpdate(){};
    virtual void update() {};
    virtual frontend::FrontendResource get_resource() { return {}; };
};

using EntryPointExecutionCallback =
    std::function<bool(
          void*
        , std::vector<megamol::frontend::FrontendResource> const&
        , ImageWrapper&
        )>;

struct EntryPoint {
    std::string moduleName;
    void* modulePtr = nullptr;
    std::vector<megamol::frontend::FrontendResource> entry_point_resources;
    // pimpl to some implementation handling rendering input data
    std::unique_ptr<RenderInputsUpdate> entry_point_data = std::make_unique<RenderInputsUpdate>();

    EntryPointExecutionCallback execute;
    ImageWrapper execution_result_image;
};

} /* end namespace frontend_resources */
} /* end namespace megamol */
