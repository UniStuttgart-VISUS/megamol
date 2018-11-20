/*
 * AbstractBezierRenderer.cpp
 *
 * Copyright (C) 2010 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#define _USE_MATH_DEFINES
#include "vislib/graphics/gl/IncludeAllGL.h"
#include "AbstractBezierRenderer.h"
#include "mmcore/AbstractGetData3DCall.h"


namespace megamol {
namespace demos {


/*
 * AbstractBezierRenderer::AbstractBezierRenderer
 */
AbstractBezierRenderer::AbstractBezierRenderer(void) : Renderer3DModule(),
        getDataSlot("getdata", "Connects to the data source"),
        objsHash(0), shader(NULL), scaling(1.0f) {
    // intentionally empty
}


/*
 * AbstractBezierRenderer::~AbstractBezierRenderer
 */
AbstractBezierRenderer::~AbstractBezierRenderer(void) {
    this->Release();
}


/*
 * AbstractBezierRenderer::create
 */
bool AbstractBezierRenderer::create(void) {
    if (!vislib::graphics::gl::GLSLShader::InitialiseExtensions()) return false;
    return true;
}


/*
 * AbstractBezierRenderer::GetExtents
 */
bool AbstractBezierRenderer::GetExtents(core::Call& call) {
    core::view::CallRender3D *cr = dynamic_cast<core::view::CallRender3D*>(&call);
    if (cr == NULL) return false;

    core::AbstractGetData3DCall *gd = this->getDataSlot.CallAs<core::AbstractGetData3DCall>();
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
 * AbstractBezierRenderer::release
 */
void AbstractBezierRenderer::release(void) {
    this->shader = NULL; // Do not release or delete ...
}


/*
 * AbstractBezierRenderer::Render
 */
bool AbstractBezierRenderer::Render(core::Call& call) {
    core::view::CallRender3D *cr = dynamic_cast<core::view::CallRender3D*>(&call);
    if (cr == NULL) return false;
    // As sfx 'this->scaling' has already been set! :-)

    if (this->shader_required() && (this->shader == NULL)) return false;
    return this->render(*cr);
}

} /* end namespace demos */
} /* end namespace megamol */