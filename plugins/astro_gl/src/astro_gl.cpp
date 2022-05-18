/**
 * MegaMol
 * Copyright (c) 2009-2021, MegaMol Dev Team
 * All rights reserved.
 */

#include "mmcore/utility/plugins/AbstractPluginInstance.h"
#include "mmcore/utility/plugins/PluginRegister.h"

#include "SurfaceLICRenderer.h"


namespace megamol::astro_gl {
class AstroGLPluginInstance : public megamol::core::utility::plugins::AbstractPluginInstance {
    REGISTERPLUGIN(AstroGLPluginInstance)

public:
    AstroGLPluginInstance()
            : megamol::core::utility::plugins::AbstractPluginInstance("astro_gl", "The astro plugin."){};

    ~AstroGLPluginInstance() override = default;

    // Registers modules and calls
    void registerClasses() override {

        // register modules
        this->module_descriptions.RegisterAutoDescription<megamol::astro_gl::SurfaceLICRenderer>();

        // register calls
    }
};
} // namespace megamol::astro_gl
