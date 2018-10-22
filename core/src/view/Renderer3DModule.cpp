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
    this->renderSlot.SetCallback(CallRender3D::ClassName(),
        CallRender3D::FunctionName(CallRender3D::FnGetCapabilities),
		&Renderer3DModule::GetCapabilitiesCallback);
    this->MakeSlotAvailable(&this->renderSlot);
}

/*
 * view::Renderer3DModule::OnMouseEventCallback
 */
bool Renderer3DModule::GetCapabilitiesCallback(Call& call) {
    try {
        view::CallRender3D& cr = dynamic_cast<view::CallRender3D&>(call);
        return this->GetCapabilities(cr);
    } catch (...) {
        ASSERT("GetCapabilitiesCallback call cast failed\n");
    }
    return false;
}
