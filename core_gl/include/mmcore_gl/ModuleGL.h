/*
 * Module.h
 *
 * Copyright (C) 2021 by VISUS (Universitaet Stuttgart).
 * Alle Rechte vorbehalten.
 */

#pragma once

#include "mmcore/Module.h"

namespace megamol {
namespace core_gl {

/**
 * Base class of all graph modules
 */
class ModuleGL : public core::Module {
public:
    std::vector<std::string> requested_lifetime_resources() override {
        std::vector<std::string> resources = Module::requested_lifetime_resources();
        resources.emplace_back("OpenGL_Context"); // GL modules should request the GL context resource
        return resources;
    }
};


} /* end namespace core_gl */
} /* end namespace megamol */
