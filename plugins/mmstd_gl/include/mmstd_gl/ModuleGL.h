/**
 * MegaMol
 * Copyright (c) 2021, MegaMol Dev Team
 * All rights reserved.
 */

#pragma once

#include "OpenGL_Context.h"
#include "RuntimeConfig.h"
#include "mmcore/Module.h"

namespace megamol::mmstd_gl {

/**
 * Base class of all graph modules
 */
class ModuleGL : public core::Module {
public:
    static void requested_lifetime_resources(frontend_resources::ResourceRequest& req) {
        Module::requested_lifetime_resources(req);
        req.require<frontend_resources::OpenGL_Context>(); // GL modules should request the GL context resource
        req.require<frontend_resources::RuntimeConfig>();  // GL modules probably need shader paths
    }
};

} // namespace megamol::mmstd_gl
