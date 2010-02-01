/*
 * SolPathRenderer.cpp
 *
 * Copyright (C) 2010 by VISUS (University of Stuttgart)
 * Alle Rechte vorbehalten.
 */
#include "stdafx.h"
#include "SolPathRenderer.h"
#include "CallAutoDescription.h"
#include "CoreInstance.h"
#include "SolPathDataCall.h"
#include "utility/ShaderSourceFactory.h"
#include "view/CallRender3D.h"
#include "vislib/ShaderSource.h"
#include <GL/gl.h>

using namespace megamol;
using namespace megamol::protein;


/*
 * SolPathRenderer::SolPathRenderer
 */
SolPathRenderer::SolPathRenderer(void) : core::view::Renderer3DModule(),
        getdataslot("getdata", "Fetches data"), pathlineShader() {

    this->getdataslot.SetCompatibleCall<core::CallAutoDescription<SolPathDataCall> >();
    this->MakeSlotAvailable(&this->getdataslot);
}


/*
 * SolPathRenderer::~SolPathRenderer
 */
SolPathRenderer::~SolPathRenderer(void) {
    this->Release();
}


/*
 * SolPathRenderer::create
 */
bool SolPathRenderer::create(void) {
    using vislib::sys::Log;
    if (!vislib::graphics::gl::GLSLShader::InitialiseExtensions()) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
            "Unable to initialise opengl extensions for glsl");
        return false;
    }

    try {
        vislib::graphics::gl::ShaderSource vertSrc;
        vislib::graphics::gl::ShaderSource fragSrc;

        if (!this->GetCoreInstance()->ShaderSourceFactory().MakeShaderSource("solpath::pathline::vert", vertSrc)) {
            Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "Unable to load pathline vertex shader source");
            return false;
        }
        if (!this->GetCoreInstance()->ShaderSourceFactory().MakeShaderSource("solpath::pathline::frag", fragSrc)) {
            Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "Unable to load pathline fragment shader source");
            return false;
        }

        if (!this->pathlineShader.Create(vertSrc.Code(), vertSrc.Count(), fragSrc.Code(), fragSrc.Count())) {
            Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "Unable to create pathline shader");
            return false;
        }

    } catch(vislib::Exception e) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "Unable to create pathline shader: %s", e.GetMsgA());
        return false;
    } catch(...) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "Unable to create pathline shader");
        return false;
    }

    return true;
}


/*
 * SolPathRenderer::GetCapabilities
 */
bool SolPathRenderer::GetCapabilities(core::Call& call) {
    core::view::CallRender3D *cr3d = dynamic_cast<core::view::CallRender3D*>(&call);
    if (cr3d == NULL) return false;

    cr3d->SetCapabilities(
        core::view::CallRender3D::CAP_RENDER
        | core::view::CallRender3D::CAP_LIGHTING);

    return true;
}


/*
 * SolPathRenderer::GetExtents
 */
bool SolPathRenderer::GetExtents(core::Call& call) {
    core::view::CallRender3D *cr3d = dynamic_cast<core::view::CallRender3D*>(&call);
    if (cr3d == NULL) return false;

    SolPathDataCall *spdc = this->getdataslot.CallAs<SolPathDataCall>();
    if (spdc == NULL) return false;

    (*spdc)(1); // get extents

    cr3d->AccessBoundingBoxes() = spdc->AccessBoundingBoxes();

    return true;
}


/*
 * SolPathRenderer::release
 */
void SolPathRenderer::release(void) {
    this->pathlineShader.Release();
}


/*
 * SolPathRenderer::Render
 */
bool SolPathRenderer::Render(core::Call& call) {
    core::view::CallRender3D *cr3d = dynamic_cast<core::view::CallRender3D*>(&call);
    if (cr3d == NULL) return false;

    SolPathDataCall *spdc = this->getdataslot.CallAs<SolPathDataCall>();
    if (spdc == NULL) return false;

    (*spdc)(0); // get data

    ::glLineWidth(1.0f);
    ::glDisable(GL_LINE_SMOOTH);
    ::glDisable(GL_BLEND);
    ::glEnable(GL_DEPTH_TEST);
    ::glPointSize(2.0f);
    ::glEnable(GL_POINT_SMOOTH);

    this->pathlineShader.Enable();

    ::glColor3ub(192, 192, 192);
    const SolPathDataCall::Pathline *path = spdc->Pathlines();
    for (unsigned int p = 0; p < spdc->Count(); p++, path++) {
        ::glBegin(GL_LINE_STRIP);
        for (unsigned int v = 0; v < path->length; v++) {
            ::glVertex3fv(&path->data[v].x);
        }
        ::glEnd();
    }

    this->pathlineShader.Disable();

    ::glPointSize(2.0f);
    ::glEnable(GL_POINT_SMOOTH);
    ::glEnable(GL_BLEND);
    ::glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    ::glColor3ub(255, 0, 0);
    path = spdc->Pathlines();
    for (unsigned int p = 0; p < spdc->Count(); p++, path++) {
        ::glBegin(GL_POINTS);
        for (unsigned int v = 0; v < path->length; v++) {
            ::glVertex3fv(&path->data[v].x);
        }
        ::glEnd();
    }
    ::glDisable(GL_POINT_SMOOTH);

    return false;
}
