/*
 * flowvis.cpp
 * Copyright (C) 2018-2021 by MegaMol Team
 * Alle Rechte vorbehalten.
 */

#include "mmcore/utility/plugins/AbstractPluginInstance.h"
#include "mmcore/utility/plugins/PluginRegister.h"

#include "ExtractPores.h"

namespace megamol::flowvis {
/** Implementing the instance class of this plugin */
class FlowvisPluginInstance : public megamol::core::utility::plugins::AbstractPluginInstance {
    REGISTERPLUGIN(FlowvisPluginInstance)

public:
    FlowvisPluginInstance()
            : megamol::core::utility::plugins::AbstractPluginInstance("flowvis", "Plugin for flow visualization."){};

    ~FlowvisPluginInstance() override = default;

    // Registers modules and calls
    void registerClasses() override {

        // register modules
        this->module_descriptions.RegisterAutoDescription<megamol::flowvis::ExtractPores>();

        // register calls
        
    }
};
} // namespace megamol::flowvis
