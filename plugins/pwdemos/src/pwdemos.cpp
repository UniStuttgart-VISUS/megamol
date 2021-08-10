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

#include "AOSphereRenderer.h"
#include "BezierCPUMeshRenderer.h"
#include "CrystalStructureVolumeRenderer.h"
#include "PoreNetExtractor.h"
#include "QuartzCrystalDataSource.h"
#include "QuartzCrystalRenderer.h"
#include "QuartzDataGridder.h"
#include "QuartzParticleFortLoader.h"
#include "QuartzPlaneRenderer.h"
#include "QuartzPlaneTexRenderer.h"
#include "QuartzRenderer.h"
#include "QuartzTexRenderer.h"

#include "QuartzCrystalDataCall.h"
#include "QuartzParticleDataCall.h"
#include "QuartzParticleGridDataCall.h"

namespace megamol::demos {
    class PwdemosPluginInstance : public megamol::core::utility::plugins::AbstractPluginInstance {
        REGISTERPLUGIN(PwdemosPluginInstance)

    public:
        PwdemosPluginInstance()
                : megamol::core::utility::plugins::AbstractPluginInstance("pwdemos", "The pwdemos plugin."){};

        ~PwdemosPluginInstance() override = default;

        // Registers modules and calls
        void registerClasses() override {

            // register modules
            this->module_descriptions.RegisterAutoDescription<megamol::demos::AOSphereRenderer>();
            this->module_descriptions.RegisterAutoDescription<megamol::demos::BezierCPUMeshRenderer>();
            this->module_descriptions.RegisterAutoDescription<megamol::demos::CrystalDataSource>();
            this->module_descriptions.RegisterAutoDescription<megamol::demos::CrystalRenderer>();
            this->module_descriptions.RegisterAutoDescription<megamol::demos::QuartzPlaneRenderer>();
            this->module_descriptions.RegisterAutoDescription<megamol::demos::QuartzPlaneTexRenderer>();
            this->module_descriptions.RegisterAutoDescription<megamol::demos::QuartzRenderer>();
            this->module_descriptions.RegisterAutoDescription<megamol::demos::DataGridder>();
            this->module_descriptions.RegisterAutoDescription<megamol::demos::ParticleFortLoader>();
            this->module_descriptions.RegisterAutoDescription<megamol::demos::QuartzTexRenderer>();
            this->module_descriptions.RegisterAutoDescription<megamol::demos::PoreNetExtractor>();


#if (defined(WITH_CUDA) && (WITH_CUDA))
            this->module_descriptions.RegisterAutoDescription<megamol::demos::CrystalStructureVolumeRenderer>();
#endif

            // register calls
            this->call_descriptions.RegisterAutoDescription<megamol::demos::CrystalDataCall>();
            this->call_descriptions.RegisterAutoDescription<megamol::demos::ParticleDataCall>();
            this->call_descriptions.RegisterAutoDescription<megamol::demos::ParticleGridDataCall>();
        }
    };
} // namespace megamol::demos
