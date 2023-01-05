/**
 * MegaMol
 * Copyright (c) 2009-2021, MegaMol Dev Team
 * All rights reserved.
 */

#include "mmcore/factories/AbstractPluginInstance.h"
#include "mmcore/factories/PluginRegister.h"

#include "MapGenerator.h"

namespace megamol::molecularmaps {

class MolecularmapsPluginInstance : public megamol::core::factories::AbstractPluginInstance {
    REGISTERPLUGIN(MolecularmapsPluginInstance)

public:
    MolecularmapsPluginInstance()
            : megamol::core::factories::AbstractPluginInstance(
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
