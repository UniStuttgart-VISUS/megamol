/**
 * MegaMol
 * Copyright (c) 2022, MegaMol Dev Team
 * All rights reserved.
 */

#include "mmcore/utility/plugins/AbstractPluginInstance.h"
#include "mmcore/utility/plugins/PluginRegister.h"

namespace megamol::mmstd {
class PluginInstance : public megamol::core::utility::plugins::AbstractPluginInstance {
    REGISTERPLUGIN(PluginInstance)
public:
    PluginInstance() : megamol::core::utility::plugins::AbstractPluginInstance("mmstd", "Core calls and modules."){};

    ~PluginInstance() override = default;

    // Registers modules and calls
    void registerClasses() override {

        // register modules

        // register calls
    }
};
} // namespace megamol::mmstd
