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
class MEGAMOLCORE_API ModuleGL : public core::Module {
public:
    std::vector<std::string> requested_lifetime_resources() override { 
        return
        {
            "GlobalValueStore"
            , "IOpenGL_Context" // request for this resource may be deleted any time - this is just an example. but GL modules should request the GL context resource.
        };
    }
};


} /* end namespace core_gl */
} /* end namespace megamol */
