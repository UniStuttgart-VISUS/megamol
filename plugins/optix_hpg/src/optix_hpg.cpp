/**
 * MegaMol
 * Copyright (c) 2009-2021, MegaMol Dev Team
 * All rights reserved.
 */

#include "mmcore/utility/plugins/AbstractPluginInstance.h"
#include "mmcore/utility/plugins/PluginRegister.h"

#include "CUDAToGL.h"
#include "optix/MeshGeometry.h"
#include "optix/Renderer.h"
#include "optix/SphereGeometry.h"
#include "optix/TransitionCalculator.h"

#include "CallRender3DCUDA.h"
#include "optix/CallGeometry.h"

namespace megamol::optix_hpg {
class OptixHpgPluginInstance : public megamol::core::utility::plugins::AbstractPluginInstance {
    REGISTERPLUGIN(OptixHpgPluginInstance)

public:
    OptixHpgPluginInstance(void)
            : megamol::core::utility::plugins::AbstractPluginInstance("optix_hpg", "The optix_hpg plugin."){};

    ~OptixHpgPluginInstance() override = default;

    // Registers modules and calls
    void registerClasses() override {

        // register modules
        this->module_descriptions.RegisterAutoDescription<megamol::optix_hpg::SphereGeometry>();
        this->module_descriptions.RegisterAutoDescription<megamol::optix_hpg::Renderer>();
        this->module_descriptions.RegisterAutoDescription<megamol::optix_hpg::MeshGeometry>();
        this->module_descriptions.RegisterAutoDescription<megamol::optix_hpg::TransitionCalculator>();
        this->module_descriptions.RegisterAutoDescription<megamol::optix_hpg::CUDAToGL>();

        // register calls
        this->call_descriptions.RegisterAutoDescription<megamol::optix_hpg::CallGeometry>();
        this->call_descriptions.RegisterAutoDescription<megamol::optix_hpg::CallRender3DCUDA>();
    }
};
} // namespace megamol::optix_hpg
