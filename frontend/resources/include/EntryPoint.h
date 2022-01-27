/*
 * EntryPoint.h
 *
 * Copyright (C) 2021 by VISUS (Universitaet Stuttgart).
 * Alle Rechte vorbehalten.
 */

#pragma once

#include "FrontendResource.h"
#include "ImageWrapper.h"

namespace megamol {
namespace frontend_resources {

// a way for entry points to get new render input state from outside
// update of the render input gets called before the entry point is rendered/executed
struct RenderInputsUpdate {
    virtual ~RenderInputsUpdate(){};
    virtual void update(){};
    virtual frontend::FrontendResource get_resource() {
        return {};
    };
};

// the function/callback that gets executed when an entry point should get rendered
using EntryPointExecutionCallback = std::function<bool(void* /*module ptr, e.g. view*/,
    std::vector<megamol::frontend::FrontendResource> const& /*input: requested resources*/,
    ImageWrapper& /*output: rendering result*/)>;

// generic structure to encapsulate "renderable" things that can be rendered by the ImagePresentation Service
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
