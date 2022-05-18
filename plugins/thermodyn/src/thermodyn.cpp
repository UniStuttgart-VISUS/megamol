/**
 * MegaMol
 * Copyright (c) 2009-2021, MegaMol Dev Team
 * All rights reserved.
 */

#include "mmcore/utility/plugins/AbstractPluginInstance.h"
#include "mmcore/utility/plugins/PluginRegister.h"

#include "PhaseAnimator.h"
#include "PhaseSeparator.h"

#include "thermodyn/BoxDataCall.h"

namespace megamol::thermodyn {
class ThermodynPluginInstance : public megamol::core::utility::plugins::AbstractPluginInstance {
    REGISTERPLUGIN(ThermodynPluginInstance)

public:
    ThermodynPluginInstance()
            : megamol::core::utility::plugins::AbstractPluginInstance("thermodyn", "The thermodyn plugin."){};

    ~ThermodynPluginInstance() override = default;

    // Registers modules and calls
    void registerClasses() override {

        // register modules
        this->module_descriptions.RegisterAutoDescription<megamol::thermodyn::PhaseSeparator>();
        this->module_descriptions.RegisterAutoDescription<megamol::thermodyn::PhaseAnimator>();

        // register calls
        this->call_descriptions.RegisterAutoDescription<megamol::thermodyn::BoxDataCall>();
    }
};
} // namespace megamol::thermodyn
