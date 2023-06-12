/**
 * MegaMol
 * Copyright (c) 2023, MegaMol Dev Team
 * All rights reserved.
 */

#include "mmcore/factories/AbstractPluginInstance.h"
#include "mmcore/factories/PluginRegister.h"

#include "rendering/SRTest.h"

namespace megamol::MegaMolPlugin {
class PluginInstance : public megamol::core::factories::AbstractPluginInstance {
    REGISTERPLUGIN(PluginInstance)
public:
    PluginInstance()
            : megamol::core::factories::AbstractPluginInstance(
                  // machine-readable plugin assembly name
                  "Test_GL",

                  // human-readable plugin description
                  "Collection of modules for testing"){};

    ~PluginInstance() override = default;

    // Registers modules and calls
    void registerClasses() override {

        // register modules
        this->module_descriptions.RegisterAutoDescription<megamol::test_gl::rendering::SRTest>();

        // register calls
    }
};
} // namespace megamol::MegaMolPlugin
