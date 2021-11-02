/*
 * CallClassRegistry.cpp
 * Copyright (C) 2008 - 2015 by MegaMol Consortium
 * All rights reserved. Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "mmcore_gl/factories/CallClassRegistryGL.h"

#include "mmcore/factories/CallDescriptionManager.h"
#include "mmcore/factories/CallAutoDescription.h"
#include "mmcore/factories/CallDescription.h"

#include "mmcore_gl/view/CallGetTransferFunctionGL.h"
#include "mmcore_gl/view/CallRender2DGL.h"
#include "mmcore_gl/view/CallRender3DGL.h"
#include "mmcore_gl/view/CallRenderViewGL.h"
#include "mmcore_gl/UniFlagCallsGL.h"
#include "mmcore_gl/view/special/CallbackScreenShooter.h"


/*
 * factories::register_call_classes
 */
void megamol::core_gl::factories::register_call_classes_gl(megamol::core::factories::CallDescriptionManager& instance) {
    //////////////////////////////////////////////////////////////////////
    // Register all rendering graph call descriptions here
    //////////////////////////////////////////////////////////////////////
    instance.RegisterAutoDescription<core_gl::view::CallGetTransferFunctionGL>();
    instance.RegisterAutoDescription<core_gl::view::CallRender2DGL>();
    instance.RegisterAutoDescription<core_gl::view::CallRender3DGL>();
    instance.RegisterAutoDescription<core_gl::view::CallRenderViewGL>();
    instance.RegisterAutoDescription<core_gl::view::special::CallbackScreenShooterCall>();
    instance.RegisterAutoDescription<core_gl::FlagCallRead_GL>();
    instance.RegisterAutoDescription<core_gl::FlagCallWrite_GL>();
}
