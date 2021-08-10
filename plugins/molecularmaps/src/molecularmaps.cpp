/**
 * MegaMol
 * Copyright (c) 2009-2021, MegaMol Dev Team
 * All rights reserved.
 */

// TODO: Vislib must die!!!
// Vislib includes Windows.h. This crashes when somebody else (i.e. zmq) is using Winsock2.h, but the vislib include
// is first without defining WIN32_LEAN_AND_MEAN. This define is the only thing we need from stdafx.h, include could be
// removed otherwise.
#include "stdafx.h"

#include "mmcore/utility/plugins/AbstractPluginInstance.h"
#include "mmcore/utility/plugins/PluginRegister.h"

#include "MapGenerator.h"

namespace megamol::molecularmaps {

class MolecularmapsPluginInstance : public megamol::core::utility::plugins::AbstractPluginInstance {
    REGISTERPLUGIN(MolecularmapsPluginInstance)

public:
    MolecularmapsPluginInstance()
            : megamol::core::utility::plugins::AbstractPluginInstance(
                  "molecularmaps", "New version of the molecular maps creator"){};

    ~MolecularmapsPluginInstance() override = default;

    // Registers modules and calls
    void registerClasses() override {

        // register modules
        this->module_descriptions.RegisterAutoDescription<megamol::molecularmaps::MapGenerator>();

        // register calls
    }
};
} // namespace megamol::molecularmaps
