/*
 * Renderer3DModule.cpp
 *
 * Copyright (C) 2008 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "mmcore/view/Renderer3DModule.h"

using namespace megamol::core;
using namespace megamol::core::view;

/*
 * view::Renderer3DModule::Renderer3DModule
 */
Renderer3DModule::Renderer3DModule() : RendererModule<CallRender3D>() {
    this->MakeSlotAvailable(&this->renderSlot);
}
