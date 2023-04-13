/**
 * MegaMol
 * Copyright (c) 2021, MegaMol Dev Team
 * All rights reserved.
 */

#include "mmcore/factories/AbstractPluginInstance.h"
#include "mmcore/factories/PluginRegister.h"

namespace megamol::MegaMolPlugin {
class PluginInstance : public megamol::core::factories::AbstractPluginInstance {
    REGISTERPLUGIN(PluginInstance)
public:
    PluginInstance()
            : megamol::core::factories::AbstractPluginInstance(
                  // machine-readable plugin assembly name
                  "MegaMolPlugin", // TODO: Change this!

                  // human-readable plugin description
                  "Describing MegaMolPlugin (TODO: Change this!)"){};

    ~PluginInstance() override = default;

    // Registers modules and calls
    void registerClasses() override {

        // register modules

        //
        // TODO: Register your plugin's modules here:
        // this->module_descriptions.RegisterAutoDescription<megamol::MegaMolPlugin::MyModule1>();
        // this->module_descriptions.RegisterAutoDescription<megamol::MegaMolPlugin::MyModule2>();
        // ...
        //

        // register calls

        // TODO: Register your plugin's calls here:
        // this->call_descriptions.RegisterAutoDescription<megamol::MegaMolPlugin::MyCall1>();
        // this->call_descriptions.RegisterAutoDescription<megamol::MegaMolPlugin::MyCall2>();
        // ...
        //
    }
};
} // namespace megamol::MegaMolPlugin
