/*
 * AbstractBezierRaycastRenderer.cpp
 *
 * Copyright (C) 2010 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#define _USE_MATH_DEFINES
#include "glh/glh_extensions.h"
#include "AbstractBezierRaycastRenderer.h"
#include "AbstractGetData3DCall.h"
//#include "BezierDataCall.h"
//#include "param/EnumParam.h"
//#include "param/IntParam.h"
//#include <cmath>

using namespace megamol::core;


/*
 * misc::AbstractBezierRaycastRenderer::AbstractBezierRaycastRenderer
 */
misc::AbstractBezierRaycastRenderer::AbstractBezierRaycastRenderer(void) : Renderer3DModule(),
        getDataSlot("getdata", "Connects to the data source"),
        objsHash(0), shader(NULL), scaling(1.0f) {
    // intentionally empty
}


/*
 * misc::AbstractBezierRaycastRenderer::~AbstractBezierRaycastRenderer
 */
misc::AbstractBezierRaycastRenderer::~AbstractBezierRaycastRenderer(void) {
    this->Release();
}


/*
 * misc::AbstractBezierRaycastRenderer::create
 */
bool misc::AbstractBezierRaycastRenderer::create(void) {
    if (!vislib::graphics::gl::GLSLShader::InitialiseExtensions()) return false;
    return true;
}


/*
 * misc::AbstractBezierRaycastRenderer::GetCapabilities
 */
bool misc::AbstractBezierRaycastRenderer::GetCapabilities(Call& call) {
    view::CallRender3D *cr = dynamic_cast<view::CallRender3D*>(&call);
    if (cr == NULL) return false;

    cr->SetCapabilities(
        view::CallRender3D::CAP_RENDER
        | view::CallRender3D::CAP_LIGHTING
        );

    return true;
}


/*
 * misc::AbstractBezierRaycastRenderer::GetExtents
 */
bool misc::AbstractBezierRaycastRenderer::GetExtents(Call& call) {
    view::CallRender3D *cr = dynamic_cast<view::CallRender3D*>(&call);
    if (cr == NULL) return false;

    AbstractGetData3DCall *gd = this->getDataSlot.CallAs<AbstractGetData3DCall>();
    if ((gd != NULL) && ((*gd)(1))) {
        cr->SetTimeFramesCount(gd->FrameCount());
        cr->AccessBoundingBoxes() = gd->AccessBoundingBoxes();
        if (cr->AccessBoundingBoxes().IsObjectSpaceBBoxValid()) {
            this->scaling = cr->AccessBoundingBoxes().ObjectSpaceBBox().LongestEdge();
            if (this->scaling > 0.0001f) {
                this->scaling = 2.0f / this->scaling;
            } else {
                this->scaling = 2.0f;
            }
            cr->AccessBoundingBoxes().MakeScaledWorld(this->scaling);
        }

    } else {
        cr->SetTimeFramesCount(1);
        cr->AccessBoundingBoxes().Clear();
        this->scaling = 1.0f;
    }

    return true;
}


/*
 * misc::AbstractBezierRaycastRenderer::release
 */
void misc::AbstractBezierRaycastRenderer::release(void) {
    this->shader = NULL; // Do not release or delete ...
}


/*
 * misc::AbstractBezierRaycastRenderer::Render
 */
bool misc::AbstractBezierRaycastRenderer::Render(Call& call) {
    view::CallRender3D *cr = dynamic_cast<view::CallRender3D*>(&call);
    if (cr == NULL) return false;
    // As sfx 'this->scaling' has already been set! :-)

    if (this->shader == NULL) return false;
    return this->render(*cr);
}
