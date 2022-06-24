/*
 * ClassRegistry.cpp
 * Copyright (C) 2021 by MegaMol Consortium
 * All rights reserved. Alle Rechte vorbehalten.
 */

#include "mmcore/utility/plugins/AbstractPluginInstance.h"
#include "mmcore/utility/plugins/PluginRegister.h"

#include "mmcore_gl/DeferredShading.h"
#include "mmcore_gl/flags/UniFlagStorage.h"
#include "mmcore_gl/view/BoundingBoxRenderer.h"
#include "mmcore_gl/view/ContextToGL.h"
#include "mmcore_gl/view/HeadView.h"
#include "mmcore_gl/view/SplitViewGL.h"
#include "mmcore_gl/view/TransferFunctionGL.h"
#include "mmcore_gl/view/View2DGL.h"
#include "mmcore_gl/view/View3DGL.h"
#include "mmcore_gl/view/special/CallbackScreenShooter.h"
#include "mmcore_gl/view/special/ChronoGraph.h"
#include "mmcore_gl/view/special/DemoRenderer2D.h"
#include "mmcore_gl/view/special/ScreenShooter.h"

#include "mmcore_gl/flags/FlagCallsGL.h"
#include "mmcore_gl/view/CallGetTransferFunctionGL.h"
#include "mmcore_gl/view/CallRender2DGL.h"
#include "mmcore_gl/view/CallRender3DGL.h"
#include "mmcore_gl/view/CallRenderViewGL.h"


namespace megamol::core_gl {
class CoreGLPluginInstance : public megamol::core::utility::plugins::AbstractPluginInstance {
    REGISTERPLUGIN(CoreGLPluginInstance)

public:
    CoreGLPluginInstance()
            : megamol::core::utility::plugins::AbstractPluginInstance("core_gl", "Everything GL from the core"){};

    ~CoreGLPluginInstance() override = default;

    // Registers modules and calls
    void registerClasses() override {

        //////////////////////////////////////////////////////////////////////
        // Register all module descriptions here
        //////////////////////////////////////////////////////////////////////

        this->module_descriptions.RegisterAutoDescription<core_gl::view::TransferFunctionGL>();
        this->module_descriptions.RegisterAutoDescription<core_gl::view::special::ChronoGraph>();
        this->module_descriptions.RegisterAutoDescription<core_gl::view::special::DemoRenderer2D>();
        this->module_descriptions.RegisterAutoDescription<core_gl::view::special::ScreenShooter>();
        this->module_descriptions.RegisterAutoDescription<core_gl::view::View2DGL>();
        this->module_descriptions.RegisterAutoDescription<core_gl::view::View3DGL>();
        this->module_descriptions.RegisterAutoDescription<core_gl::view::HeadView>();
        this->module_descriptions.RegisterAutoDescription<core_gl::view::BoundingBoxRenderer>();
        this->module_descriptions.RegisterAutoDescription<core_gl::view::SplitViewGL>();
        this->module_descriptions.RegisterAutoDescription<core_gl::view::special::CallbackScreenShooter>();
        this->module_descriptions.RegisterAutoDescription<core_gl::DeferredShading>();
        this->module_descriptions.RegisterAutoDescription<core_gl::UniFlagStorage>();

        //////////////////////////////////////////////////////////////////////
        // Register all call descriptions here
        //////////////////////////////////////////////////////////////////////

        this->call_descriptions.RegisterAutoDescription<core_gl::view::CallGetTransferFunctionGL>();
        this->call_descriptions.RegisterAutoDescription<core_gl::view::CallRender2DGL>();
        this->call_descriptions.RegisterAutoDescription<core_gl::view::CallRender3DGL>();
        this->call_descriptions.RegisterAutoDescription<core_gl::view::CallRenderViewGL>();
        this->call_descriptions.RegisterAutoDescription<core_gl::view::special::CallbackScreenShooterCall>();
        this->call_descriptions.RegisterAutoDescription<core_gl::FlagCallRead_GL>();
        this->call_descriptions.RegisterAutoDescription<core_gl::FlagCallWrite_GL>();
    }
};
} // namespace megamol::core_gl
