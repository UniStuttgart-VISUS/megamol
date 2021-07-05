/*
 * thermodyn.cpp
 * Copyright (C) 2009-2015 by MegaMol Team
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"

#include "mmcore/api/MegaMolCore.std.h"
#include "mmcore/utility/plugins/Plugin200Instance.h"
#include "mmcore/utility/plugins/PluginRegister.h"
#include "mmcore/versioninfo.h"
#include "vislib/vislibversion.h"

#include "PhaseSeparator.h"
#include "PhaseAnimator.h"
#include "rendering/BoxRenderer.h"

#include "thermodyn/BoxDataCall.h"

namespace megamol::thermodyn {
/** Implementing the instance class of this plugin */
class plugin_instance : public ::megamol::core::utility::plugins::Plugin200Instance {
    REGISTERPLUGIN(plugin_instance)
public:
    /** ctor */
    plugin_instance(void)
        : ::megamol::core::utility::plugins::Plugin200Instance(

              /* machine-readable plugin assembly name */
              "thermodyn", // TODO: Change this!

              /* human-readable plugin description */
              "Describing thermodyn (TODO: Change this!)"){

              // here we could perform addition initialization
          };
    /** Dtor */
    virtual ~plugin_instance(void) {
        // here we could perform addition de-initialization
    }
    /** Registers modules and calls */
    virtual void registerClasses(void) {

        // register modules here:
        this->module_descriptions.RegisterAutoDescription<megamol::thermodyn::PhaseSeparator>();
        this->module_descriptions.RegisterAutoDescription<megamol::thermodyn::PhaseAnimator>();
        this->module_descriptions.RegisterAutoDescription<megamol::thermodyn::rendering::BoxRenderer>();
        //
        // TODO: Register your plugin's modules here
        // like:
        //   this->module_descriptions.RegisterAutoDescription<megamol::thermodyn::MyModule1>();
        //   this->module_descriptions.RegisterAutoDescription<megamol::thermodyn::MyModule2>();
        //   ...
        //

        // register calls here:
        this->call_descriptions.RegisterAutoDescription<megamol::thermodyn::BoxDataCall>();
        //
        // TODO: Register your plugin's calls here
        // like:
        //   this->call_descriptions.RegisterAutoDescription<megamol::thermodyn::MyCall1>();
        //   this->call_descriptions.RegisterAutoDescription<megamol::thermodyn::MyCall2>();
        //   ...
        //
    }
};
} // namespace megamol::thermodyn
