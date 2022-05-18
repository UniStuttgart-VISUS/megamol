/**
 * MegaMol
 * Copyright (c) 2009-2021, MegaMol Dev Team
 * All rights reserved.
 */

#include "mmcore/utility/plugins/AbstractPluginInstance.h"
#include "mmcore/utility/plugins/PluginRegister.h"

#include "OSPRayToGL.h"

namespace megamol::ospray_gl {
class MMOSPRayGLPluginInstance : public megamol::core::utility::plugins::AbstractPluginInstance {
    REGISTERPLUGIN(MMOSPRayGLPluginInstance)

public:
    MMOSPRayGLPluginInstance(void)
            : megamol::core::utility::plugins::AbstractPluginInstance("mmospray_gl", "CPU Raytracing"){};

    ~MMOSPRayGLPluginInstance() override = default;

    // Registers modules and calls
    void registerClasses() override {

        // register modules
        this->module_descriptions.RegisterAutoDescription<megamol::ospray_gl::OSPRayToGL>();

        // register calls
    }
};
} // namespace megamol::ospray_gl
