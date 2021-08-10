/**
 * MegaMol
 * Copyright (c) 2009-2021, MegaMol Dev Team
 * All rights reserved.
 */

// TODO: Vislib must die!!!
// Vislib includes Windows.h. This crashes when somebody else (i.e. zmq) is using Winsock2.h, but the vislib include
// is first without defining WIN32_LEAN_AND_MEAN. This define is the only thing we need from stdafx.h, include could be
// removed otherwise.
#include "stdafx.h"

#include "mmcore/utility/plugins/AbstractPluginInstance.h"
#include "mmcore/utility/plugins/PluginRegister.h"

#include "AstroParticleConverter.h"
#include "AstroSchulz.h"
#include "Contest2019DataLoader.h"
#include "DirectionToColour.h"
#include "FilamentFilter.h"
#include "SimpleAstroFilter.h"
#include "SpectralIntensityVolume.h"
#include "SurfaceLICRenderer.h"
#include "VolumetricGlobalMinMax.h"
#include "astro/AstroDataCall.h"

namespace megamol::astro {
    class AstroPluginInstance : public megamol::core::utility::plugins::AbstractPluginInstance {
        REGISTERPLUGIN(AstroPluginInstance)

    public:
        AstroPluginInstance() : megamol::core::utility::plugins::AbstractPluginInstance("astro", "The astro plugin."){};

        ~AstroPluginInstance() override = default;

        // Registers modules and calls
        void registerClasses() override {

            // register modules
            this->module_descriptions.RegisterAutoDescription<megamol::astro::Contest2019DataLoader>();
            this->module_descriptions.RegisterAutoDescription<megamol::astro::AstroParticleConverter>();
            this->module_descriptions.RegisterAutoDescription<megamol::astro::FilamentFilter>();
            this->module_descriptions.RegisterAutoDescription<megamol::astro::AstroSchulz>();
            this->module_descriptions.RegisterAutoDescription<megamol::astro::DirectionToColour>();
            this->module_descriptions.RegisterAutoDescription<megamol::astro::SimpleAstroFilter>();
            this->module_descriptions.RegisterAutoDescription<megamol::astro::SurfaceLICRenderer>();
            this->module_descriptions.RegisterAutoDescription<megamol::astro::SpectralIntensityVolume>();
            this->module_descriptions.RegisterAutoDescription<megamol::astro::VolumetricGlobalMinMax>();

            // register calls
            this->call_descriptions.RegisterAutoDescription<megamol::astro::AstroDataCall>();
        }
    };
} // namespace megamol::astro
