/*
 * AbstractBezierRenderer.cpp
 *
 * Copyright (C) 2010 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#define _USE_MATH_DEFINES
#include "vislib_gl/graphics/gl/IncludeAllGL.h"
#include "AbstractBezierRenderer.h"
#include "mmcore/AbstractGetData3DCall.h"

#include "OpenGL_Context.h"


namespace megamol {
namespace demos_gl {


/*
 * AbstractBezierRenderer::AbstractBezierRenderer
 */
AbstractBezierRenderer::AbstractBezierRenderer(void) : core_gl::view::Renderer3DModuleGL(),
        getDataSlot("getdata", "Connects to the data source"),
        objsHash(0), shader(NULL) {
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
    auto const& ogl_ctx = frontend_resources.get<frontend_resources::OpenGL_Context>();
    if (!ogl_ctx.areExtAvailable(vislib_gl::graphics::gl::GLSLShader::RequiredExtensions()))
        return false;

    return true;
}


/*
 * AbstractBezierRenderer::GetExtents
 */
bool AbstractBezierRenderer::GetExtents(core_gl::view::CallRender3DGL& call) {

    core::AbstractGetData3DCall *gd = this->getDataSlot.CallAs<core::AbstractGetData3DCall>();
    if ((gd != NULL) && ((*gd)(1))) {
        call.SetTimeFramesCount(gd->FrameCount());
        call.AccessBoundingBoxes() = gd->AccessBoundingBoxes();
    } else {
        call.SetTimeFramesCount(1);
        call.AccessBoundingBoxes().Clear();
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
bool AbstractBezierRenderer::Render(core_gl::view::CallRender3DGL& call) {

    if (this->shader_required() && (this->shader == NULL)) return false;
    return this->render(call);
}

} /* end namespace demos */
} /* end namespace megamol */
