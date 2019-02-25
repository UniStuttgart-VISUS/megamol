/*
 * BoundingBoxRenderer.cpp
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */
#include "stdafx.h"
#include "mmcore/nextgen/BoundingBoxRenderer.h"

#include "mmcore/param/BoolParam.h"
#include "mmcore/param/ColorParam.h"

using namespace megamol::core;
using namespace megamol::core::nextgen;

BoundingBoxRenderer::BoundingBoxRenderer(void)
    : core::nextgen::Renderer3DModule_2()
    , enableBoundingBoxSlot("enableBoundingBox", "Enables the rendering of the bounding box")
    , boundingBoxColorSlot("boundingBoxColor", "Color of the bounding box")
    , enableViewCubeSlot("enableViewCube", "Enables the rendering of the view cube") {

    this->enableBoundingBoxSlot.SetParameter(new param::BoolParam(true));
    this->MakeSlotAvailable(&this->enableBoundingBoxSlot);

    this->boundingBoxColorSlot.SetParameter(new param::ColorParam("#ffffffff"));
    this->MakeSlotAvailable(&this->boundingBoxColorSlot);

    this->enableViewCubeSlot.SetParameter(new param::BoolParam(false));
    this->MakeSlotAvailable(&this->enableViewCubeSlot);
}

BoundingBoxRenderer::~BoundingBoxRenderer(void) {
    this->Release();
}

bool BoundingBoxRenderer::create(void) {

    return true;
}

void BoundingBoxRenderer::release(void) {

}

bool BoundingBoxRenderer::GetExtents(CallRender3D_2& call) {

    return true;
}

bool BoundingBoxRenderer::Render(CallRender3D_2& call) {

    return true;
}

bool BoundingBoxRenderer::RenderBoundingBox(CallRender3D_2& call) {

    return true;
}

bool BoundingBoxRenderer::RenderViewCube(CallRender3D_2& call) {

    return true;
}