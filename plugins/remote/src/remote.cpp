/**
 * MegaMol
 * Copyright (c) 2009-2021, MegaMol Dev Team
 * All rights reserved.
 */

#include "mmcore/factories/AbstractPluginInstance.h"
#include "mmcore/factories/PluginRegister.h"

// Modules
#include "FBOCompositor2.h"
#include "FBOTransmitter2.h"
#include "HeadnodeServer.h"
#include "RendernodeView.h"

// Calls

namespace megamol::remote {
class RemotePluginInstance : public megamol::core::factories::AbstractPluginInstance {
    REGISTERPLUGIN(RemotePluginInstance)

public:
    RemotePluginInstance(void)
            : megamol::core::factories::AbstractPluginInstance(
                  "remote", "Plugin containing remote utilities for MegaMol"){};

    ~RemotePluginInstance() override = default;

    // Registers modules and calls
    void registerClasses() override {

        // register modules
        this->module_descriptions.RegisterAutoDescription<megamol::remote::FBOTransmitter2>();
        this->module_descriptions.RegisterAutoDescription<megamol::remote::FBOCompositor2>();
        this->module_descriptions.RegisterAutoDescription<megamol::remote::HeadnodeServer>();
        this->module_descriptions.RegisterAutoDescription<megamol::remote::RendernodeView>();

        // register calls
    }
};
} // namespace megamol::remote
