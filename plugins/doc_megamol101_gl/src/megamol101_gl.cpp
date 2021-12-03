/**
 * MegaMol
 * Copyright (c) 2016, MegaMol Dev Team
 * All rights reserved.
 */

#include "mmcore/utility/plugins/AbstractPluginInstance.h"
#include "mmcore/utility/plugins/PluginRegister.h"

#include "ASCIISphereLoader.h"
#include "CallSpheres.h"
#include "SimplestSphereRenderer.h"
#include "SphereColoringModule.h"

namespace megamol::megamol101_gl {
class Megamol101GLPluginInstance : public megamol::core::utility::plugins::AbstractPluginInstance {
    REGISTERPLUGIN(Megamol101GLPluginInstance)

public:
    Megamol101GLPluginInstance()
            : megamol::core::utility::plugins::AbstractPluginInstance("megamol101_gl", "The megamol101 plugin."){};

    ~Megamol101GLPluginInstance() override = default;

    // Registers modules and calls
    void registerClasses() override {

        // register modules
        this->module_descriptions.RegisterAutoDescription<megamol::megamol101_gl::ASCIISphereLoader>();
        this->module_descriptions.RegisterAutoDescription<megamol::megamol101_gl::SimplestSphereRenderer>();
        this->module_descriptions.RegisterAutoDescription<megamol::megamol101_gl::SphereColoringModule>();

        // register calls
        this->call_descriptions.RegisterAutoDescription<megamol::megamol101_gl::CallSpheres>();
    }
};
} // namespace megamol::megamol101_gl
