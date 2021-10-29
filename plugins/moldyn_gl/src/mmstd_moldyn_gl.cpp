/**
 * MegaMol
 * Copyright (c) 2009-2021, MegaMol Dev Team
 * All rights reserved.
 */

#include "mmcore/utility/plugins/AbstractPluginInstance.h"
#include "mmcore/utility/plugins/PluginRegister.h"

#include "rendering/ArrowRenderer.h"
#include "rendering/GlyphRenderer.h"
#include "rendering/GrimRenderer.h"
#include "rendering/SphereRenderer.h"

namespace megamol::stdplugin::moldyn_gl {
    class MoldynGLPluginInstance : public megamol::core::utility::plugins::AbstractPluginInstance {
    REGISTERPLUGIN(MoldynGLPluginInstance)

    public:
    MoldynGLPluginInstance()
                : megamol::core::utility::plugins::AbstractPluginInstance(
                      "mmstd_moldyn_gl", "MegaMol Plugins for Molecular Dynamics Data Visualization"){};

        ~MoldynGLPluginInstance() override = default;

        // Registers modules and calls
        void registerClasses() override {

            // register modules
            this->module_descriptions.RegisterAutoDescription<megamol::stdplugin::moldyn_gl::rendering::GrimRenderer>();
            this->module_descriptions.RegisterAutoDescription<megamol::stdplugin::moldyn_gl::rendering::ArrowRenderer>();
            this->module_descriptions.RegisterAutoDescription<megamol::stdplugin::moldyn_gl::rendering::SphereRenderer>();
            this->module_descriptions.RegisterAutoDescription<megamol::stdplugin::moldyn_gl::rendering::GlyphRenderer>();
        }
    };
} // namespace megamol::stdplugin::moldyn
