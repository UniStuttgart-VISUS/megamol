/**
 * MegaMol
 * Copyright (c) 2021, MegaMol Dev Team
 * All rights reserved.
 */

#include "mmcore/utility/plugins/Plugin200Instance.h"
#include "mmcore/utility/plugins/PluginRegister.h"

namespace megamol::MegaMolPlugin {
    class PluginInstance : public ::megamol::core::utility::plugins::Plugin200Instance {
        REGISTERPLUGIN(PluginInstance)
    public:
        PluginInstance()
                : ::megamol::core::utility::plugins::Plugin200Instance(

                      // machine-readable plugin assembly name
                      "MegaMolPlugin", // TODO: Change this!

                      // human-readable plugin description
                      "Describing MegaMolPlugin (TODO: Change this!)"){

                      // here we could perform addition initialization
                  };

        ~PluginInstance() override {
            // here we could perform addition de-initialization
        }

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
