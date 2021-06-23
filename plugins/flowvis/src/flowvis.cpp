/*
 * flowvis.cpp
 * Copyright (C) 2018-2019 by MegaMol Team
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"

#include "mmcore/api/MegaMolCore.std.h"
#include "mmcore/utility/plugins/Plugin200Instance.h"
#include "mmcore/utility/plugins/PluginRegister.h"
#include "mmcore/versioninfo.h"
#include "vislib/vislibversion.h"

#include "ExtractPores.h"

namespace megamol::flowvis {
/** Implementing the instance class of this plugin */
class plugin_instance : public ::megamol::core::utility::plugins::Plugin200Instance {
    REGISTERPLUGIN(plugin_instance)
public:
    /** ctor */
    plugin_instance(void)
        : ::megamol::core::utility::plugins::Plugin200Instance(

              /* machine-readable plugin assembly name */
              "flowvis",

              /* human-readable plugin description */
              "Plugin for flow visualization."){

              // here we could perform addition initialization
          };
    /** Dtor */
    virtual ~plugin_instance(void) {
        // here we could perform addition de-initialization
    }
    /** Registers modules and calls */
    virtual void registerClasses(void) {

        // register modules here:
        this->module_descriptions.RegisterAutoDescription<megamol::flowvis::ExtractPores>();

        // register calls here:
        
    }
};
} // namespace megamol::flowvis
