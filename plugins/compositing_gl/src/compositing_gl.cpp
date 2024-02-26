/**
 * MegaMol
 * Copyright (c) 2009-2021, MegaMol Dev Team
 * All rights reserved.
 */

#include "mmcore/factories/AbstractPluginInstance.h"
#include "mmcore/factories/PluginRegister.h"

#include "AntiAliasing.h"
#include "DepthDarkening.h"
#include "DrawToScreen.h"
#include "InteractionRenderTarget.h"
#include "LocalLighting.h"
#include "NormalFromDepth.h"
#include "OpenEXRReader.h"
#include "OpenEXRWriter.h"
#include "PNGDataSource.h"
#include "SSAO.h"
#include "ScreenSpaceEdges.h"
#include "SimpleRenderTarget.h"
#include "TexInspectModule.h"
#include "TextureCombine.h"
#include "TextureDepthCompositing.h"
#include "compositing_gl/CompositingCalls.h"

namespace megamol::compositing_gl {
class CompositingPluginInstance : public megamol::core::factories::AbstractPluginInstance {
    REGISTERPLUGIN(CompositingPluginInstance)

public:
    CompositingPluginInstance()
            : megamol::core::factories::AbstractPluginInstance("compositing_gl", "The compositing_gl plugin."){};

    ~CompositingPluginInstance() override = default;

    // Registers modules and calls
    void registerClasses() override {

        // register modules
        this->module_descriptions.RegisterAutoDescription<megamol::compositing_gl::DepthDarkening>();
        this->module_descriptions.RegisterAutoDescription<megamol::compositing_gl::DrawToScreen>();
        this->module_descriptions.RegisterAutoDescription<megamol::compositing_gl::InteractionRenderTarget>();
        this->module_descriptions.RegisterAutoDescription<megamol::compositing_gl::LocalLighting>();
        this->module_descriptions.RegisterAutoDescription<megamol::compositing_gl::SimpleRenderTarget>();
        this->module_descriptions.RegisterAutoDescription<megamol::compositing_gl::TextureCombine>();
        this->module_descriptions.RegisterAutoDescription<megamol::compositing_gl::TextureDepthCompositing>();
        this->module_descriptions.RegisterAutoDescription<megamol::compositing_gl::NormalFromDepth>();
        this->module_descriptions.RegisterAutoDescription<megamol::compositing_gl::SSAO>();
        this->module_descriptions.RegisterAutoDescription<megamol::compositing_gl::AntiAliasing>();
        this->module_descriptions.RegisterAutoDescription<megamol::compositing_gl::PNGDataSource>();
        this->module_descriptions.RegisterAutoDescription<megamol::compositing_gl::TexInspectModule>();
        this->module_descriptions.RegisterAutoDescription<megamol::compositing_gl::ScreenSpaceEdges>();
        this->module_descriptions.RegisterAutoDescription<megamol::compositing_gl::OpenEXRWriter>();
        this->module_descriptions.RegisterAutoDescription<megamol::compositing_gl::OpenEXRReader>();

        // register calls
        this->call_descriptions.RegisterAutoDescription<megamol::compositing_gl::CallTexture2D>();
        this->call_descriptions.RegisterAutoDescription<megamol::compositing_gl::CallCamera>();
        this->call_descriptions.RegisterAutoDescription<megamol::compositing_gl::CallFramebufferGL>();
        this->call_descriptions.RegisterAutoDescription<megamol::compositing_gl::CallTextureFormat>();
    }
};
} // namespace megamol::compositing_gl
