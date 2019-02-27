/*
 * BoundingBoxRenderer.cpp
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */
#include "stdafx.h"
#include "mmcore/nextgen/BoundingBoxRenderer.h"

#include "vislib/sys/Log.h"
#include "mmcore/param/BoolParam.h"
#include "mmcore/param/ColorParam.h"

using namespace megamol::core;
using namespace megamol::core::nextgen;
using namespace megamol::core::view;

/*
 * BoundingBoxRenderer::BoundingBoxRenderer
 */
BoundingBoxRenderer::BoundingBoxRenderer(void)
    : RendererModule<CallRender3D_2>()
    , enableBoundingBoxSlot("enableBoundingBox", "Enables the rendering of the bounding box")
    , boundingBoxColorSlot("boundingBoxColor", "Color of the bounding box")
    , enableViewCubeSlot("enableViewCube", "Enables the rendering of the view cube") {

    this->enableBoundingBoxSlot.SetParameter(new param::BoolParam(true));
    this->MakeSlotAvailable(&this->enableBoundingBoxSlot);

    this->boundingBoxColorSlot.SetParameter(new param::ColorParam("#ffffffff"));
    this->MakeSlotAvailable(&this->boundingBoxColorSlot);

    this->enableViewCubeSlot.SetParameter(new param::BoolParam(false));
    this->MakeSlotAvailable(&this->enableViewCubeSlot);

    this->MakeSlotAvailable(&this->chainRenderSlot);
    this->MakeSlotAvailable(&this->renderSlot);
}

/*
 * BoundingBoxRenderer::~BoundingBoxRenderer
 */
BoundingBoxRenderer::~BoundingBoxRenderer(void) { 
    this->Release();
}

/*
 * BoundingBoxRenderer::create
 */
bool BoundingBoxRenderer::create(void) { 
    // TODO shaders
    return true; 
}

/*
 * BoundingBoxRenderer::release
 */
void BoundingBoxRenderer::release(void) {
    // TODO
}

/*
 * BoundingBoxRenderer::GetExtentsChain
 */
bool BoundingBoxRenderer::GetExtentsChain(CallRender3D_2& call) {
    CallRender3D_2 * chainedCall = this->chainRenderSlot.CallAs<CallRender3D_2>();
    if (chainedCall == nullptr) {
        vislib::sys::Log::DefaultLog.WriteError("The BoundingBoxRenderer does not work without a renderer attached to its right");
        return false;
    }
    *chainedCall = call;
    bool retVal = (*chainedCall)(view::AbstractCallRender::FnGetExtents);
    call = *chainedCall;
    return retVal;
}

/*
 * BoundingBoxRenderer::GetExtentsChain
 */
bool BoundingBoxRenderer::GetExtents(CallRender3D_2& call) {
    return true;
}

/*
 * BoundingBoxRenderer::RenderChain
 */
bool BoundingBoxRenderer::RenderChain(CallRender3D_2& call) {
    CallRender3D_2 * chainedCall = this->chainRenderSlot.CallAs<CallRender3D_2>();
    if (chainedCall == nullptr) {
        vislib::sys::Log::DefaultLog.WriteError("The BoundingBoxRenderer does not work without a renderer attached to its right");
        return false;
    }

    bool renderRes = true;
    if (this->enableBoundingBoxSlot.Param<param::BoolParam>()->Value()) {
        renderRes &= this->RenderBoundingBoxBack(call);
    }
    renderRes &= (*chainedCall)(view::AbstractCallRender::FnRender);
    if (this->enableBoundingBoxSlot.Param<param::BoolParam>()->Value()) {
        renderRes &= this->RenderBoundingBoxFront(call);
    }
    if (this->enableViewCubeSlot.Param<param::BoolParam>()->Value()) {
        renderRes &= this->RenderViewCube(call);
    }

    return renderRes;
}

/*
 * BoundingBoxRenderer::Render
 */
bool BoundingBoxRenderer::Render(CallRender3D_2& call) {
    return true;
}

/*
 * BoundingBoxRenderer::RenderBoundingBoxFront
 */
bool BoundingBoxRenderer::RenderBoundingBoxFront(CallRender3D_2& call) { return true; }

/*
 * BoundingBoxRenderer::RenderBoundingBoxBack
 */
bool BoundingBoxRenderer::RenderBoundingBoxBack(CallRender3D_2& call) { return true; }

/*
 * BoundingBoxRenderer::RenderViewCube
 */
bool BoundingBoxRenderer::RenderViewCube(CallRender3D_2& call) { return true; }