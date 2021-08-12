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

namespace megamol::megamol101 {
class Megamol101PluginInstance : public megamol::core::utility::plugins::AbstractPluginInstance {
    REGISTERPLUGIN(Megamol101PluginInstance)

public:
    Megamol101PluginInstance()
            : megamol::core::utility::plugins::AbstractPluginInstance("megamol101", "The megamol101 plugin."){};

    ~Megamol101PluginInstance() override = default;

    // Registers modules and calls
    void registerClasses() override {

        // register modules
        this->module_descriptions.RegisterAutoDescription<megamol::megamol101::ASCIISphereLoader>();
        this->module_descriptions.RegisterAutoDescription<megamol::megamol101::SimplestSphereRenderer>();
        this->module_descriptions.RegisterAutoDescription<megamol::megamol101::SphereColoringModule>();

        // register calls
        this->call_descriptions.RegisterAutoDescription<megamol::megamol101::CallSpheres>();
    }
};
} // namespace megamol::megamol101
