/*
 * probe_gl.cpp
 * Copyright (C) 2009-2015 by MegaMol Team
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"

#include "mmcore/api/MegaMolCore.std.h"
#include "mmcore/utility/plugins/Plugin200Instance.h"
#include "mmcore/utility/plugins/PluginRegister.h"
#include "mmcore/versioninfo.h"
#include "vislib/vislibversion.h"

#include "ProbeBillboardGlyphMaterial.h"
#include "ProbeBillboardGlyphRenderTasks.h"
#include "ProbeInteraction.h"
#include "ProbeRenderTasks.h"
#include "ProbeGlCalls.h"

namespace megamol::probe_gl {
/** Implementing the instance class of this plugin */
class plugin_instance : public ::megamol::core::utility::plugins::Plugin200Instance {
    REGISTERPLUGIN(plugin_instance)
public:
    /** ctor */
    plugin_instance(void)
        : ::megamol::core::utility::plugins::Plugin200Instance(

              /* machine-readable plugin assembly name */
              "probe_gl", // TODO: Change this!

              /* human-readable plugin description */
              "Describing probe_gl (TODO: Change this!)"){

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
        //   this->module_descriptions.RegisterAutoDescription<megamol::probe_gl::MyModule1>();
        //   this->module_descriptions.RegisterAutoDescription<megamol::probe_gl::MyModule2>();
        //   ...
        //
        this->module_descriptions.RegisterAutoDescription<megamol::probe_gl::ProbeBillboardGlyphMaterial>();
        this->module_descriptions.RegisterAutoDescription<megamol::probe_gl::ProbeBillboardGlyphRenderTasks>();
        this->module_descriptions.RegisterAutoDescription<megamol::probe_gl::ProbeRenderTasks>();
        this->module_descriptions.RegisterAutoDescription<megamol::probe_gl::ProbeInteraction>();

        // register calls here:

        //
        // TODO: Register your plugin's calls here
        // like:
        //   this->call_descriptions.RegisterAutoDescription<megamol::probe_gl::MyCall1>();
        //   this->call_descriptions.RegisterAutoDescription<megamol::probe_gl::MyCall2>();
        //   ...
        //
        this->call_descriptions.RegisterAutoDescription<megamol::probe_gl::CallProbeInteraction>();
    }
};
} // namespace megamol::probe_gl
