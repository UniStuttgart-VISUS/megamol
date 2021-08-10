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

#include "PhaseAnimator.h"
#include "PhaseSeparator.h"
#include "rendering/BoxRenderer.h"

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
        this->module_descriptions.RegisterAutoDescription<megamol::thermodyn::rendering::BoxRenderer>();

        // register calls
        this->call_descriptions.RegisterAutoDescription<megamol::thermodyn::BoxDataCall>();
    }
};
} // namespace megamol::thermodyn
