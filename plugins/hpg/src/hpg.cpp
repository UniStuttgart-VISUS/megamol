/*
 * hpg.cpp
 * Copyright (C) 2009-2015 by MegaMol Team
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "hpg/hpg.h"

#include "mmcore/api/MegaMolCore.std.h"
#include "mmcore/utility/plugins/Plugin200Instance.h"
#include "mmcore/utility/plugins/PluginRegister.h"
#include "mmcore/versioninfo.h"
#include "vislib/vislibversion.h"

#include "hpg/optix/CallGeometry.h"
#include "hpg/optix/CallContext.h"

#include "optix/Renderer.h"
#include "optix/SphereGeometry.h"
#include "optix/Context.h"
#include "optix/MeshGeometry.h"
#include "optix/TransitionCalculator.h"


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
                "hpg", // TODO: Change this!

                /* human-readable plugin description */
                "Describing hpg (TODO: Change this!)") {

            // here we could perform addition initialization
        };
        /** Dtor */
        virtual ~plugin_instance(void) {
            // here we could perform addition de-initialization
        }
        /** Registers modules and calls */
        virtual void registerClasses(void) {

            // register modules here:
            this->module_descriptions.RegisterAutoDescription<megamol::hpg::optix::Renderer>();
            this->module_descriptions.RegisterAutoDescription<megamol::hpg::optix::SphereGeometry>();
            this->module_descriptions.RegisterAutoDescription<megamol::hpg::optix::Context>();
            this->module_descriptions.RegisterAutoDescription<megamol::hpg::optix::MeshGeometry>();
            this->module_descriptions.RegisterAutoDescription<megamol::hpg::optix::TransitionCalculator>();
            //
            // TODO: Register your plugin's modules here
            // like:
            //   this->module_descriptions.RegisterAutoDescription<megamol::hpg::MyModule1>();
            //   this->module_descriptions.RegisterAutoDescription<megamol::hpg::MyModule2>();
            //   ...
            //

            // register calls here:
            this->call_descriptions.RegisterAutoDescription<megamol::hpg::optix::CallGeometry>();
            this->call_descriptions.RegisterAutoDescription<megamol::hpg::optix::CallContext>();
            //
            // TODO: Register your plugin's calls here
            // like:
            //   this->call_descriptions.RegisterAutoDescription<megamol::hpg::MyCall1>();
            //   this->call_descriptions.RegisterAutoDescription<megamol::hpg::MyCall2>();
            //   ...
            //

        }
    };
}
