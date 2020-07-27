/*
 * remote.cpp
 * Copyright (C) 2009-2015 by MegaMol Team
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"

#include "mmcore/api/MegaMolCore.std.h"
#include "mmcore/utility/plugins/Plugin200Instance.h"
#include "mmcore/utility/plugins/PluginRegister.h"
#include "mmcore/versioninfo.h"
#include "vislib/vislibversion.h"

// Modules
#include "FBOCompositor2.h"
#include "FBOTransmitter2.h"
#include "HeadnodeServer.h"
#include "RendernodeView.h"

// Calls

namespace megamol::remote {
/** Implementing the instance class of this plugin */
class plugin_instance : public ::megamol::core::utility::plugins::Plugin200Instance {
    REGISTERPLUGIN(plugin_instance)
public:
    /** ctor */
    plugin_instance(void)
        : ::megamol::core::utility::plugins::Plugin200Instance(

              /* machine-readable plugin assembly name */
              "remote", // TODO: Change this!

              /* human-readable plugin description */
              "Plugin containing remote utilities for MegaMol"){

              // here we could perform addition initialization
          };
    /** Dtor */
    virtual ~plugin_instance(void) {
        // here we could perform addition de-initialization
    }
    /** Registers modules and calls */
    virtual void registerClasses(void) {

        // register modules here:
        this->module_descriptions.RegisterAutoDescription<megamol::remote::FBOTransmitter2>();
        this->module_descriptions.RegisterAutoDescription<megamol::remote::FBOCompositor2>();
        this->module_descriptions.RegisterAutoDescription<megamol::remote::HeadnodeServer>();
        this->module_descriptions.RegisterAutoDescription<megamol::remote::RendernodeView>();

		//
        // TODO: Register your plugin's modules here
        // like:
        //   this->module_descriptions.RegisterAutoDescription<megamol::remote::MyModule1>();
        //   this->module_descriptions.RegisterAutoDescription<megamol::remote::MyModule2>();
        //   ...
        //

        // register calls here:
        //
        // TODO: Register your plugin's calls here
        // like:
        //   this->call_descriptions.RegisterAutoDescription<megamol::remote::MyCall1>();
        //   this->call_descriptions.RegisterAutoDescription<megamol::remote::MyCall2>();
        //   ...
        //
    }
};
} // namespace megamol::remote
