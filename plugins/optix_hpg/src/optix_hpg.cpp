/*
 * optix_hpg.cpp
 * Copyright (C) 2009-2015 by MegaMol Team
 * Alle Rechte vorbehalten.
 */

#include "mmcore/api/MegaMolCore.std.h"
#include "mmcore/utility/plugins/Plugin200Instance.h"
#include "mmcore/utility/plugins/PluginRegister.h"
#include "mmcore/versioninfo.h"
#include "vislib/vislibversion.h"

#include "optix/SphereGeometry.h"
#include "optix/Renderer.h"
#include "optix/MeshGeometry.h"
#include "optix/TransitionCalculator.h"
#include "CUDAToGL.h"

#include "optix/CallGeometry.h"
#include "CallRender3DCUDA.h"

namespace megamol::optix_hpg {
    /** Implementing the instance class of this plugin */
    class plugin_instance : public ::megamol::core::utility::plugins::Plugin200Instance {
        REGISTERPLUGIN(plugin_instance)
    public:
        /** ctor */
        plugin_instance(void)
            : ::megamol::core::utility::plugins::Plugin200Instance(

                /* machine-readable plugin assembly name */
                "optix_hpg", // TODO: Change this!

                /* human-readable plugin description */
                "Describing optix_hpg (TODO: Change this!)") {

            // here we could perform addition initialization
        };
        /** Dtor */
        virtual ~plugin_instance(void) {
            // here we could perform addition de-initialization
        }
        /** Registers modules and calls */
        virtual void registerClasses(void) {

            // register modules here:
            this->module_descriptions.RegisterAutoDescription<megamol::optix_hpg::SphereGeometry>();
            this->module_descriptions.RegisterAutoDescription<megamol::optix_hpg::Renderer>();
            this->module_descriptions.RegisterAutoDescription<megamol::optix_hpg::MeshGeometry>();
            this->module_descriptions.RegisterAutoDescription<megamol::optix_hpg::TransitionCalculator>();
            this->module_descriptions.RegisterAutoDescription<megamol::optix_hpg::CUDAToGL>();
            //
            // TODO: Register your plugin's modules here
            // like:
            //   this->module_descriptions.RegisterAutoDescription<megamol::optix_hpg::MyModule1>();
            //   this->module_descriptions.RegisterAutoDescription<megamol::optix_hpg::MyModule2>();
            //   ...
            //

            // register calls here:
            this->call_descriptions.RegisterAutoDescription<megamol::optix_hpg::CallGeometry>();
            this->call_descriptions.RegisterAutoDescription<megamol::optix_hpg::CallRender3DCUDA>();
            //
            // TODO: Register your plugin's calls here
            // like:
            //   this->call_descriptions.RegisterAutoDescription<megamol::optix_hpg::MyCall1>();
            //   this->call_descriptions.RegisterAutoDescription<megamol::optix_hpg::MyCall2>();
            //   ...
            //

        }
    };
} // namespace megamol::optix_hpg
