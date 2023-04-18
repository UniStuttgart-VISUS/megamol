/**
 * MegaMol
 * Copyright (c) 2009-2021, MegaMol Dev Team
 * All rights reserved.
 */

#include "mmcore/factories/AbstractPluginInstance.h"
#include "mmcore/factories/PluginRegister.h"

#include "rendering/ArrowRenderer.h"
#include "rendering/GlyphRenderer.h"
#include "rendering/GrimRenderer.h"
#include "rendering/SphereRenderer.h"
#include "rendering/VoxelGenerator.h"
#include "rendering/AO.h"

namespace megamol::moldyn_gl {
class MoldynGLPluginInstance : public megamol::core::factories::AbstractPluginInstance {
    REGISTERPLUGIN(MoldynGLPluginInstance)

public:
    MoldynGLPluginInstance()
            : megamol::core::factories::AbstractPluginInstance(
                  "moldyn_gl", "MegaMol Plugins for Molecular Dynamics Data Visualization"){};

    ~MoldynGLPluginInstance() override = default;

    // Registers modules and calls
    void registerClasses() override {

        // register modules
        this->module_descriptions.RegisterAutoDescription<megamol::moldyn_gl::rendering::GrimRenderer>();
        this->module_descriptions.RegisterAutoDescription<megamol::moldyn_gl::rendering::ArrowRenderer>();
        this->module_descriptions.RegisterAutoDescription<megamol::moldyn_gl::rendering::SphereRenderer>();
        this->module_descriptions.RegisterAutoDescription<megamol::moldyn_gl::rendering::GlyphRenderer>();
        this->module_descriptions.RegisterAutoDescription<megamol::moldyn_gl::rendering::VoxelGenerator>();
        this->module_descriptions.RegisterAutoDescription<megamol::moldyn_gl::rendering::AO>();
    }
};
} // namespace megamol::moldyn_gl
