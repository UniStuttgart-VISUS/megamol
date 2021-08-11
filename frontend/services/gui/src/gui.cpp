/*
 * gui.cpp
 * Copyright (C) 2009-2015 by MegaMol Team
 * Alle Rechte vorbehalten.
 */


#include "stdafx.h"
#include "mmcore/api/MegaMolCore.std.h"
#include "mmcore/utility/plugins/Plugin200Instance.h"
#include "mmcore/utility/plugins/PluginRegister.h"
#include "mmcore/versioninfo.h"
#include "vislib/vislibversion.h"


namespace megamol::gui {
/** Implementing the instance class of this plugin */
class plugin_instance : public ::megamol::core::utility::plugins::Plugin200Instance {
    REGISTERPLUGIN(plugin_instance)
public:
    /** ctor */
    plugin_instance()
            : ::megamol::core::utility::plugins::Plugin200Instance(

                  /* machine-readable plugin assembly name */
                  "gui",

                  /* human-readable plugin description */
                  "Graphical User Interface Plugin"){

                  // here we could perform addition initialization
              };
    /** Dtor */
    virtual ~plugin_instance() {
        // here we could perform addition de-initialization
    }
    /** Registers modules and calls */
    virtual void registerClasses() {}
};
} // namespace megamol::gui
