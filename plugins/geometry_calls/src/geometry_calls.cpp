/*
 * geometry_calls.cpp
 * Copyright (C) 2009-2015 by MegaMol Team
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"

#include "mmcore/api/MegaMolCore.std.h"
#include "mmcore/utility/plugins/Plugin200Instance.h"
#include "mmcore/utility/plugins/PluginRegister.h"
#include "mmcore/versioninfo.h"
#include "vislib/vislibversion.h"

#include "geometry_calls/CallTriMeshData.h"
#include "geometry_calls/LinesDataCall.h"

namespace megamol::geocalls {
    /** Implementing the instance class of this plugin */
    class plugin_instance : public ::megamol::core::utility::plugins::Plugin200Instance {
        REGISTERPLUGIN(plugin_instance)
    public:
        /** ctor */
        plugin_instance(void)
            : ::megamol::core::utility::plugins::Plugin200Instance(

                /* machine-readable plugin assembly name */
                "geometry_calls", // TODO: Change this!

                /* human-readable plugin description */
                "Describing geometry_calls (TODO: Change this!)") {

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
            //   this->module_descriptions.RegisterAutoDescription<megamol::geometry_calls::MyModule1>();
            //   this->module_descriptions.RegisterAutoDescription<megamol::geometry_calls::MyModule2>();
            //   ...
            //

            // register calls here:
            this->call_descriptions.RegisterAutoDescription<megamol::geocalls::CallTriMeshData>();
            this->call_descriptions.RegisterAutoDescription<megamol::geocalls::LinesDataCall>();
            //
            // TODO: Register your plugin's calls here
            // like:
            //   this->call_descriptions.RegisterAutoDescription<megamol::geometry_calls::MyCall1>();
            //   this->call_descriptions.RegisterAutoDescription<megamol::geometry_calls::MyCall2>();
            //   ...
            //

        }
    };
} // namespace megamol::geocalls
