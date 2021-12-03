/**
 * MegaMol
 * Copyright (c) 2009-2021, MegaMol Dev Team
 * All rights reserved.
 */

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

namespace megamol::demos_gl {
class PwdemosGLPluginInstance : public megamol::core::utility::plugins::AbstractPluginInstance {
    REGISTERPLUGIN(PwdemosGLPluginInstance)

public:
    PwdemosGLPluginInstance()
            : megamol::core::utility::plugins::AbstractPluginInstance("pwdemos_gl", "The pwdemos plugin."){};

    ~PwdemosGLPluginInstance() override = default;

    // Registers modules and calls
    void registerClasses() override {

        // register modules
        this->module_descriptions.RegisterAutoDescription<megamol::demos_gl::AOSphereRenderer>();
        this->module_descriptions.RegisterAutoDescription<megamol::demos_gl::BezierCPUMeshRenderer>();
        this->module_descriptions.RegisterAutoDescription<megamol::demos_gl::CrystalDataSource>();
        this->module_descriptions.RegisterAutoDescription<megamol::demos_gl::CrystalRenderer>();
        this->module_descriptions.RegisterAutoDescription<megamol::demos_gl::QuartzPlaneRenderer>();
        this->module_descriptions.RegisterAutoDescription<megamol::demos_gl::QuartzPlaneTexRenderer>();
        this->module_descriptions.RegisterAutoDescription<megamol::demos_gl::QuartzRenderer>();
        this->module_descriptions.RegisterAutoDescription<megamol::demos_gl::DataGridder>();
        this->module_descriptions.RegisterAutoDescription<megamol::demos_gl::ParticleFortLoader>();
        this->module_descriptions.RegisterAutoDescription<megamol::demos_gl::QuartzTexRenderer>();
        this->module_descriptions.RegisterAutoDescription<megamol::demos_gl::PoreNetExtractor>();


#if (defined(WITH_CUDA) && (WITH_CUDA))
        this->module_descriptions.RegisterAutoDescription<megamol::demos::CrystalStructureVolumeRenderer>();
#endif

        // register calls
        this->call_descriptions.RegisterAutoDescription<megamol::demos_gl::CrystalDataCall>();
        this->call_descriptions.RegisterAutoDescription<megamol::demos_gl::ParticleDataCall>();
        this->call_descriptions.RegisterAutoDescription<megamol::demos_gl::ParticleGridDataCall>();
    }
};
} // namespace megamol::demos_gl
