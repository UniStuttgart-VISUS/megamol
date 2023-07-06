/**
 * MegaMol
 * Copyright (c) 2009-2021, MegaMol Dev Team
 * All rights reserved.
 */

#include "mmcore/factories/AbstractPluginInstance.h"
#include "mmcore/factories/PluginRegister.h"

#include "rendering/BoxRenderer.h"

namespace megamol::thermodyn_gl {
class ThermodynGLPluginInstance : public megamol::core::factories::AbstractPluginInstance {
    REGISTERPLUGIN(ThermodynGLPluginInstance)

public:
    ThermodynGLPluginInstance()
            : megamol::core::factories::AbstractPluginInstance("thermodyn_gl", "The thermodyn plugin."){};

    ~ThermodynGLPluginInstance() override = default;

    // Registers modules and calls
    void registerClasses() override {

        // register modules
        this->module_descriptions.RegisterAutoDescription<megamol::thermodyn_gl::rendering::BoxRenderer>();

        // register calls
    }
};
} // namespace megamol::thermodyn_gl
