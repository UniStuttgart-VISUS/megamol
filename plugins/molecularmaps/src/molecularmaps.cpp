/*
 * molecularmaps.cpp
 * Copyright (C) 2009-2015 by MegaMol Team
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"


#include "MapGenerator.h"

#include "mmcore/api/MegaMolCore.std.h"
#include "mmcore/utility/plugins/Plugin200Instance.h"
#include "mmcore/utility/plugins/PluginRegister.h"
#include "mmcore/versioninfo.h"
#include "vislib/vislibversion.h"


/* anonymous namespace hides this type from any other object files */
namespace {
/** Implementing the instance class of this plugin */
class plugin_instance : public ::megamol::core::utility::plugins::Plugin200Instance {
    REGISTERPLUGIN(plugin_instance)
public:
    /** ctor */
    plugin_instance(void)
            : ::megamol::core::utility::plugins::Plugin200Instance(

                  /* machine-readable plugin assembly name */
                  "molecularmaps", // TODO: Change this!

                  /* human-readable plugin description */
                  "New version of the molecular maps creator"){

                  // here we could perform addition initialization
              };
    /** Dtor */
    virtual ~plugin_instance(void) {
        // here we could perform addition de-initialization
    }
    /** Registers modules and calls */
    virtual void registerClasses(void) {

        // register modules here:

        //
        // TODO: Register your plugin's modules here
        // like:
        //   this->module_descriptions.RegisterAutoDescription<megamol::molecularmaps::MyModule1>();
        //   this->module_descriptions.RegisterAutoDescription<megamol::molecularmaps::MyModule2>();
        //   ...
        //
        this->module_descriptions.RegisterAutoDescription<megamol::molecularmaps::MapGenerator>();

        // register calls here:

        //
        // TODO: Register your plugin's calls here
        // like:
        //   this->call_descriptions.RegisterAutoDescription<megamol::molecularmaps::MyCall1>();
        //   this->call_descriptions.RegisterAutoDescription<megamol::molecularmaps::MyCall2>();
        //   ...
        //
    }
};
} // namespace
