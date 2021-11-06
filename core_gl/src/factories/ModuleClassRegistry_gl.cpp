/*
 * ModuleClassRegistry.cpp
 * Copyright (C) 2008 - 2015 by MegaMol Consortium
 * All rights reserved. Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "mmcore_gl/factories/ModuleClassRegistryGL.h"

#include "mmcore/factories/ModuleDescriptionManager.h"
#include "mmcore/factories/ModuleAutoDescription.h"
#include "mmcore/factories/ModuleDescription.h"

#include "mmcore_gl/view/TransferFunctionGL.h"
#include "mmcore_gl/view/special/ChronoGraph.h"
#include "mmcore_gl/view/special/DemoRenderer2D.h"
#include "mmcore_gl/view/special/ScreenShooter.h"
#include "mmcore_gl/view/View2DGL.h"
#include "mmcore_gl/view/View3DGL.h"
#include "mmcore_gl/view/HeadView.h"
#include "mmcore_gl/view/BoundingBoxRenderer.h"
#include "mmcore_gl/view/SplitViewGL.h"
#include "mmcore_gl/view/special/CallbackScreenShooter.h"
#include "mmcore_gl/UniFlagStorageGL.h"
#include "mmcore_gl/DeferredShading.h"
#include "mmcore_gl/view/ContextToGL.h"


/*
 * factories::register_module_classes
 */
void megamol::core_gl::factories::register_module_classes_gl(megamol::core::factories::ModuleDescriptionManager& instance) {

    //////////////////////////////////////////////////////////////////////
    // Register all rendering graph module descriptions here
    //////////////////////////////////////////////////////////////////////

    instance.RegisterAutoDescription<core_gl::view::TransferFunctionGL>();
    instance.RegisterAutoDescription<core_gl::view::special::ChronoGraph>();
    instance.RegisterAutoDescription<core_gl::view::special::DemoRenderer2D>();
    instance.RegisterAutoDescription<core_gl::view::special::ScreenShooter>();
    instance.RegisterAutoDescription<core_gl::view::View2DGL>();
    instance.RegisterAutoDescription<core_gl::view::View3DGL>();
    instance.RegisterAutoDescription<core_gl::view::HeadView>();
    instance.RegisterAutoDescription<core_gl::view::BoundingBoxRenderer>();
    instance.RegisterAutoDescription<core_gl::view::SplitViewGL>();
    instance.RegisterAutoDescription<core_gl::view::special::CallbackScreenShooter>();
    instance.RegisterAutoDescription<core_gl::UniFlagStorageGL>();
    instance.RegisterAutoDescription<core_gl::DeferredShading>();
}
