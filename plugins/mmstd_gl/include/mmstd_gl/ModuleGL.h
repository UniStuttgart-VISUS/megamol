/**
 * MegaMol
 * Copyright (c) 2021, MegaMol Dev Team
 * All rights reserved.
 */

#pragma once

#include "RuntimeConfig.h"
#include "mmcore/Module.h"

namespace megamol::mmstd_gl {

/**
 * Base class of all graph modules
 */
class ModuleGL : public core::Module {
public:
    std::vector<std::string> requested_lifetime_resources() override {
        std::vector<std::string> resources = Module::requested_lifetime_resources();
        resources.emplace_back("OpenGL_Context"); // GL modules should request the GL context resource
        resources.emplace_back("RuntimeConfig");  // GL modules probably need shader paths
        return resources;
    }
};

} // namespace megamol::mmstd_gl
