/*
 * MegaMolPlugin.cpp
 * Copyright (C) 2009-2015 by MegaMol Team
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "MegaMolPlugin/MegaMolPlugin.h"

#include "mmcore/api/MegaMolCore.std.h"
#include "mmcore/utility/plugins/Plugin200Instance.h"
#include "mmcore/utility/plugins/PluginRegister.h"
#include "mmcore/versioninfo.h"
#include "vislib/vislibversion.h"

namespace megamol::MegaMolPlugin {
    /** Implementing the instance class of this plugin */
    class plugin_instance : public ::megamol::core::utility::plugins::Plugin200Instance {
        REGISTERPLUGIN(plugin_instance)
    public:
        /** ctor */
        plugin_instance(void)
            : ::megamol::core::utility::plugins::Plugin200Instance(

                /* machine-readable plugin assembly name */
                "MegaMolPlugin", // TODO: Change this!

                /* human-readable plugin description */
                "Describing MegaMolPlugin (TODO: Change this!)") {

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
            //   this->module_descriptions.RegisterAutoDescription<megamol::MegaMolPlugin::MyModule1>();
            //   this->module_descriptions.RegisterAutoDescription<megamol::MegaMolPlugin::MyModule2>();
            //   ...
            //

            // register calls here:

            //
            // TODO: Register your plugin's calls here
            // like:
            //   this->call_descriptions.RegisterAutoDescription<megamol::MegaMolPlugin::MyCall1>();
            //   this->call_descriptions.RegisterAutoDescription<megamol::MegaMolPlugin::MyCall2>();
            //   ...
            //

        }
    };
} // namespace megamol::MegaMolPlugin
