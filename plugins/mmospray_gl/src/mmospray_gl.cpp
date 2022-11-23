/**
 * MegaMol
 * Copyright (c) 2009-2021, MegaMol Dev Team
 * All rights reserved.
 */

#include "mmcore/factories/AbstractPluginInstance.h"
#include "mmcore/factories/PluginRegister.h"

#include "OSPRayToGL.h"

namespace megamol::ospray_gl {
class MMOSPRayGLPluginInstance : public megamol::core::factories::AbstractPluginInstance {
    REGISTERPLUGIN(MMOSPRayGLPluginInstance)

public:
    MMOSPRayGLPluginInstance(void)
            : megamol::core::factories::AbstractPluginInstance("mmospray_gl", "CPU Raytracing"){};

    ~MMOSPRayGLPluginInstance() override = default;

    // Registers modules and calls
    void registerClasses() override {

        // register modules
        this->module_descriptions.RegisterAutoDescription<megamol::ospray_gl::OSPRayToGL>();

        // register calls
    }
};
} // namespace megamol::ospray_gl
