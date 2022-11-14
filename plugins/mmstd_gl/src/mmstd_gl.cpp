/**
 * MegaMol
 * Copyright (c) 2021, MegaMol Dev Team
 * All rights reserved.
 */

#include "mmcore/utility/plugins/AbstractPluginInstance.h"
#include "mmcore/utility/plugins/PluginRegister.h"

#include "mmstd_gl/flags/FlagCallsGL.h"
#include "mmstd_gl/flags/UniFlagStorage.h"
#include "mmstd_gl/renderer/BoundingBoxRenderer.h"
#include "mmstd_gl/renderer/CallGetTransferFunctionGL.h"
#include "mmstd_gl/renderer/CallRender2DGL.h"
#include "mmstd_gl/renderer/CallRender3DGL.h"
#include "mmstd_gl/renderer/CallRenderViewGL.h"
#include "mmstd_gl/renderer/TimeMultiplier.h"
#include "mmstd_gl/renderer/TransferFunctionGL.h"
#include "mmstd_gl/special/CallbackScreenShooter.h"
#include "mmstd_gl/special/ChronoGraph.h"
#include "mmstd_gl/special/DemoRenderer2D.h"
#include "mmstd_gl/special/ScreenShooter.h"
#include "mmstd_gl/view/HeadView.h"
#include "mmstd_gl/view/SplitViewGL.h"
#include "mmstd_gl/view/View2DGL.h"
#include "mmstd_gl/view/View3DGL.h"
#include "renderer/PlaneRenderer.h"
#include "upscaling/ImageSpaceAmortization2D.h"
#include "upscaling/ResolutionScaler2D.h"
#include "upscaling/ResolutionScaler3D.h"

namespace megamol::mmstd_gl {
class PluginInstance : public megamol::core::utility::plugins::AbstractPluginInstance {
    REGISTERPLUGIN(PluginInstance)
public:
    PluginInstance()
            : megamol::core::utility::plugins::AbstractPluginInstance("mmstd_gl", "CoreGL calls and modules."){};

    ~PluginInstance() override = default;

    // Registers modules and calls
    void registerClasses() override {

        // register modules
        this->module_descriptions.RegisterAutoDescription<mmstd_gl::view::View2DGL>();
        this->module_descriptions.RegisterAutoDescription<mmstd_gl::view::View3DGL>();
        this->module_descriptions.RegisterAutoDescription<mmstd_gl::view::SplitViewGL>();
        this->module_descriptions.RegisterAutoDescription<mmstd_gl::view::HeadView>();
        this->module_descriptions.RegisterAutoDescription<mmstd_gl::TransferFunctionGL>();
        this->module_descriptions.RegisterAutoDescription<mmstd_gl::BoundingBoxRenderer>();
        this->module_descriptions.RegisterAutoDescription<mmstd_gl::PlaneRenderer>();
        this->module_descriptions.RegisterAutoDescription<mmstd_gl::UniFlagStorage>();
        this->module_descriptions.RegisterAutoDescription<mmstd_gl::ImageSpaceAmortization2D>();
        this->module_descriptions.RegisterAutoDescription<mmstd_gl::ResolutionScaler2D>();
        this->module_descriptions.RegisterAutoDescription<mmstd_gl::ResolutionScaler3D>();
        this->module_descriptions.RegisterAutoDescription<mmstd_gl::TimeMultiplier>();
        this->module_descriptions.RegisterAutoDescription<mmstd_gl::special::ScreenShooter>();
        this->module_descriptions.RegisterAutoDescription<mmstd_gl::special::CallbackScreenShooter>();
        this->module_descriptions.RegisterAutoDescription<mmstd_gl::special::ChronoGraph>();
        this->module_descriptions.RegisterAutoDescription<mmstd_gl::special::DemoRenderer2D>();

        // register calls
        this->call_descriptions.RegisterAutoDescription<mmstd_gl::CallRenderViewGL>();
        this->call_descriptions.RegisterAutoDescription<mmstd_gl::CallRender2DGL>();
        this->call_descriptions.RegisterAutoDescription<mmstd_gl::CallRender3DGL>();
        this->call_descriptions.RegisterAutoDescription<mmstd_gl::CallGetTransferFunctionGL>();
        this->call_descriptions.RegisterAutoDescription<mmstd_gl::FlagCallRead_GL>();
        this->call_descriptions.RegisterAutoDescription<mmstd_gl::FlagCallWrite_GL>();
        this->call_descriptions.RegisterAutoDescription<mmstd_gl::special::CallbackScreenShooterCall>();
    }
};
} // namespace megamol::mmstd_gl
