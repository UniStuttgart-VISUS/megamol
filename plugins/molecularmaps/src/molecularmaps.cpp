/**
 * MegaMol
 * Copyright (c) 2009-2021, MegaMol Dev Team
 * All rights reserved.
 */

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
