/*
 * pwdemos.cpp
 * Copyright (C) 2009-2015 by MegaMol Team
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"

#include "mmcore/api/MegaMolCore.std.h"
#include "mmcore/utility/plugins/Plugin200Instance.h"
#include "mmcore/utility/plugins/PluginRegister.h"
#include "mmcore/versioninfo.h"
#include "vislib/vislibversion.h"

#include "AOSphereRenderer.h"
#include "BezierCPUMeshRenderer.h"
#include "QuartzCrystalDataSource.h"
#include "CrystalStructureVolumeRenderer.h"
#include "QuartzCrystalRenderer.h"
#include "QuartzDataGridder.h"
#include "QuartzRenderer.h"
#include "QuartzParticleFortLoader.h"
#include "QuartzTexRenderer.h"
#include "QuartzPlaneTexRenderer.h"
#include "QuartzPlaneRenderer.h"
#include "PoreNetExtractor.h"

#include "QuartzCrystalDataCall.h"
#include "QuartzParticleDataCall.h"
#include "QuartzParticleGridDataCall.h"

namespace megamol::demos {
    /** Implementing the instance class of this plugin */
    class plugin_instance : public ::megamol::core::utility::plugins::Plugin200Instance {
        REGISTERPLUGIN(plugin_instance)
    public:
        /** ctor */
        plugin_instance(void)
            : ::megamol::core::utility::plugins::Plugin200Instance(

                /* machine-readable plugin assembly name */
                "pwdemos", // TODO: Change this!

                /* human-readable plugin description */
                "Describing pwdemos (TODO: Change this!)") {

            // here we could perform addition initialization
        };
        /** Dtor */
        virtual ~plugin_instance(void) {
            // here we could perform addition de-initialization
        }
        /** Registers modules and calls */
        virtual void registerClasses(void) {

            // register modules here:

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

            // register calls here:

            this->call_descriptions.RegisterAutoDescription<megamol::demos::CrystalDataCall>();
            this->call_descriptions.RegisterAutoDescription<megamol::demos::ParticleDataCall>();
            this->call_descriptions.RegisterAutoDescription<megamol::demos::ParticleGridDataCall>();

        }
    };
} // namespace megamol::demos
