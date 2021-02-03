/*
 * astro.cpp
 * Copyright (C) 2009-2015 by MegaMol Team
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"

#include "mmcore/api/MegaMolCore.std.h"
#include "mmcore/utility/plugins/Plugin200Instance.h"
#include "mmcore/utility/plugins/PluginRegister.h"
#include "mmcore/versioninfo.h"
#include "vislib/vislibversion.h"

#include "astro/AstroDataCall.h"
#include "AstroParticleConverter.h"
#include "AstroSchulz.h"
#include "Contest2019DataLoader.h"
#include "DirectionToColour.h"
#include "FilamentFilter.h"
#include "SimpleAstroFilter.h"
#include "SurfaceLICRenderer.h"
#include "SpectralIntensityVolume.h"
#include "VolumetricGlobalMinMax.h"


namespace megamol::astro {
    /** Implementing the instance class of this plugin */
    class plugin_instance : public ::megamol::core::utility::plugins::Plugin200Instance {
        REGISTERPLUGIN(plugin_instance)
    public:
        /** ctor */
        plugin_instance(void)
            : ::megamol::core::utility::plugins::Plugin200Instance(

                /* machine-readable plugin assembly name */
                "astro",

                /* human-readable plugin description */
                "Describing astro (TODO: Change this!)") {

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
            //   this->module_descriptions.RegisterAutoDescription<megamol::astro::MyModule1>();
            //   this->module_descriptions.RegisterAutoDescription<megamol::astro::MyModule2>();
            //   ...
            //
            this->module_descriptions.RegisterAutoDescription<megamol::astro::Contest2019DataLoader>();
			this->module_descriptions.RegisterAutoDescription<megamol::astro::AstroParticleConverter>();
            this->module_descriptions.RegisterAutoDescription<megamol::astro::FilamentFilter>();
            this->module_descriptions.RegisterAutoDescription<megamol::astro::AstroSchulz>();
            this->module_descriptions.RegisterAutoDescription<megamol::astro::DirectionToColour>();
            this->module_descriptions.RegisterAutoDescription<megamol::astro::SimpleAstroFilter>();
            this->module_descriptions.RegisterAutoDescription<megamol::astro::SurfaceLICRenderer>();
            this->module_descriptions.RegisterAutoDescription<megamol::astro::SpectralIntensityVolume>();
            this->module_descriptions.RegisterAutoDescription<megamol::astro::VolumetricGlobalMinMax>();

            // register calls here:

            //
            // TODO: Register your plugin's calls here
            // like:
            //   this->call_descriptions.RegisterAutoDescription<megamol::astro::MyCall1>();
            //   this->call_descriptions.RegisterAutoDescription<megamol::astro::MyCall2>();
            //   ...
            //
            this->call_descriptions.RegisterAutoDescription<megamol::astro::AstroDataCall>();
        }
    };
} // namespace megamol::astro
