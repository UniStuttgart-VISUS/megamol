/**
 * MegaMol
 * Copyright (c) 2009-2021, MegaMol Dev Team
 * All rights reserved.
 */

#include "mmcore/utility/plugins/AbstractPluginInstance.h"
#include "mmcore/utility/plugins/PluginRegister.h"

#include "AntiAliasing.h"
#include "DepthDarkening.h"
#include "DrawToScreen.h"
#include "InteractionRenderTarget.h"
#include "LocalLighting.h"
#include "NormalFromDepth.h"
#include "PNGDataSource.h"
#include "SSAO.h"
#include "ScreenSpaceEffect.h"
#include "SimpleRenderTarget.h"
#include "TextureCombine.h"
#include "TextureDepthCompositing.h"
#include "compositing_gl/CompositingCalls.h"

namespace megamol::compositing {
class CompositingPluginInstance : public megamol::core::utility::plugins::AbstractPluginInstance {
    REGISTERPLUGIN(CompositingPluginInstance)

public:
    CompositingPluginInstance()
            : megamol::core::utility::plugins::AbstractPluginInstance("compositing_gl", "The compositing_gl plugin."){};

    ~CompositingPluginInstance() override = default;

    // Registers modules and calls
    void registerClasses() override {

        // register modules
        this->module_descriptions.RegisterAutoDescription<megamol::compositing::DepthDarkening>();
        this->module_descriptions.RegisterAutoDescription<megamol::compositing::DrawToScreen>();
        this->module_descriptions.RegisterAutoDescription<megamol::compositing::InteractionRenderTarget>();
        this->module_descriptions.RegisterAutoDescription<megamol::compositing::LocalLighting>();
        this->module_descriptions.RegisterAutoDescription<megamol::compositing::ScreenSpaceEffect>();
        this->module_descriptions.RegisterAutoDescription<megamol::compositing::SimpleRenderTarget>();
        this->module_descriptions.RegisterAutoDescription<megamol::compositing::TextureCombine>();
        this->module_descriptions.RegisterAutoDescription<megamol::compositing::TextureDepthCompositing>();
        this->module_descriptions.RegisterAutoDescription<megamol::compositing::NormalFromDepth>();
        this->module_descriptions.RegisterAutoDescription<megamol::compositing::SSAO>();
        this->module_descriptions.RegisterAutoDescription<megamol::compositing_gl::AntiAliasing>();
        this->module_descriptions.RegisterAutoDescription<megamol::compositing_gl::PNGDataSource>();

        // register calls
        this->call_descriptions.RegisterAutoDescription<megamol::compositing::CallTexture2D>();
        this->call_descriptions.RegisterAutoDescription<megamol::compositing::CallCamera>();
        this->call_descriptions.RegisterAutoDescription<megamol::compositing::CallFramebufferGL>();
    }
};
} // namespace megamol::compositing
