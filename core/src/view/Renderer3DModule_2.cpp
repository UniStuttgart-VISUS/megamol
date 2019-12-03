/*
 * Renderer3DModule_2.cpp
 *
 * Copyright (C) 2018 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "mmcore/view/Renderer3DModule_2.h"
#include "mmcore/view/AbstractView.h"
#include "mmcore/view/CallRender3D_2.h"
#include "mmcore/view/InputCall.h"

using namespace megamol::core;
using namespace megamol::core::view;
using namespace megamol::core::view;

/*
 * Renderer3DModule_2::Renderer3DModule_2
 */
Renderer3DModule_2::Renderer3DModule_2(void)
    : RendererModule<CallRender3D_2>()
    , lightSlot(
          "lights", "Lights are retrieved over this slot. If no light is connected, a default camera light is used") {

    // Callback should already be set by RendererModule
    this->MakeSlotAvailable(&this->chainRenderSlot);

    this->lightSlot.SetCompatibleCall<light::CallLightDescription>();
    this->MakeSlotAvailable(&this->lightSlot);

    // Callback should already be set by RendererModule
    this->MakeSlotAvailable(&this->renderSlot);
}

/*
 * Renderer3DModule_2::~Renderer3DModule_2
 */
Renderer3DModule_2::~Renderer3DModule_2(void) {
    // intentionally empty
}

/*
 * Renderer3DModule_2::GetExtentsChain
 */
bool Renderer3DModule_2::GetExtentsChain(CallRender3D_2& call) {
    CallRender3D_2* chainedCall = this->chainRenderSlot.CallAs<CallRender3D_2>();
    if (chainedCall != nullptr) {
        // copy the incoming call to the output
        *chainedCall = call;

        // chain through the get extents call
        (*chainedCall)(view::AbstractCallRender::FnGetExtents);
    }

    // TODO extents magic


    // get our own extents
    this->GetExtents(call);

    if (chainedCall != nullptr) {
        auto mybb = call.AccessBoundingBoxes().BoundingBox();
        mybb.Union(chainedCall->AccessBoundingBoxes().BoundingBox());
        auto mycb = call.AccessBoundingBoxes().ClipBox();
        mycb.Union(chainedCall->AccessBoundingBoxes().ClipBox());
        call.AccessBoundingBoxes().SetBoundingBox(mybb);
        call.AccessBoundingBoxes().SetClipBox(mycb);

        // TODO machs richtig
        call.SetTimeFramesCount(chainedCall->TimeFramesCount());
    }

    return true;
}

/*
 * Renderer3DModule_2::RenderChain
 */
bool Renderer3DModule_2::RenderChain(CallRender3D_2& call) {
    auto leftSlotParent = call.PeekCallerSlot()->Parent();
    std::shared_ptr<const view::AbstractView> viewptr =
        std::dynamic_pointer_cast<const view::AbstractView>(leftSlotParent);

    if (viewptr != nullptr) {
        auto vp = call.GetViewport();
        glViewport(vp.Left(), vp.Bottom(), vp.Width(), vp.Height());
        auto backCol = call.BackgroundColor();
        glClearColor(backCol.x, backCol.y, backCol.z, 0.0f);
        glClearDepth(1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    }

	this->PreRender(call);

    CallRender3D_2* chainedCall = this->chainRenderSlot.CallAs<CallRender3D_2>();

    if (chainedCall != nullptr) {
        // copy the incoming call to the output
        *chainedCall = call;

        // chain through the render call
        (*chainedCall)(view::AbstractCallRender::FnRender);

        call = *chainedCall;
    }

    // render our own stuff
    this->Render(call);

    return true;
}

/*
 * Renderer3DModule_2::GetLights
 */
bool Renderer3DModule_2::GetLights(void) {
    core::view::light::CallLight* cl = this->lightSlot.CallAs<core::view::light::CallLight>();
    if (cl == nullptr) {
        // TODO add local light
        return false;
    }
    cl->setLightMap(&this->lightMap);
    cl->fillLightMap();
    bool lightDirty = false;
    for (const auto element : this->lightMap) {
        auto light = element.second;
        if (light.dataChanged) {
            lightDirty = true;
        }
    }
    return lightDirty;
}

/*
 * Renderer3DModule_2::PreRender
 */
void Renderer3DModule_2::PreRender(CallRender3D_2& call) {
	//intentionally empty
}