/*
 * SolPathRenderer.cpp
 *
 * Copyright (C) 2010 by VISUS (University of Stuttgart)
 * Alle Rechte vorbehalten.
 */
#include "stdafx.h"

#include "SolPathRenderer.h"
#include "mmcore/factories/CallAutoDescription.h"
#include "mmcore/CoreInstance.h"
#include "SolPathDataCall.h"
#include "mmcore/utility/ShaderSourceFactory.h"
#include "mmcore/view/CallRender3D.h"
#include "vislib/math/mathfunctions.h"
#include "vislib/graphics/gl/ShaderSource.h"
#include "vislib/graphics/gl/IncludeAllGL.h"

using namespace megamol;
using namespace megamol::protein;


/*
 * SolPathRenderer::SolPathRenderer
 */
SolPathRenderer::SolPathRenderer(void) : core::view::Renderer3DModule(),
        getdataslot("getdata", "Fetches data"), pathlineShader() {

    this->getdataslot.SetCompatibleCall<core::factories::CallAutoDescription<SolPathDataCall> >();
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
    if (!isExtAvailable( "GL_ARB_vertex_program")  || !ogl_IsVersionGEQ(2,0)) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
            "Unable to initialise opengl extensions for ARB shaders and OGL 2.0");
        return false;
    }
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

    try {
        vislib::graphics::gl::ShaderSource vertSrc;
        vislib::graphics::gl::ShaderSource fragSrc;

        if (!this->GetCoreInstance()->ShaderSourceFactory().MakeShaderSource("solpath::dots::vert", vertSrc)) {
            Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "Unable to load dots vertex shader source");
            return false;
        }
        if (!this->GetCoreInstance()->ShaderSourceFactory().MakeShaderSource("solpath::dots::frag", fragSrc)) {
            Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "Unable to load dots fragment shader source");
            return false;
        }

        if (!this->dotsShader.Create(vertSrc.Code(), vertSrc.Count(), fragSrc.Code(), fragSrc.Count())) {
            Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "Unable to create dots shader");
            return false;
        }

    } catch(vislib::Exception e) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "Unable to create dots shader: %s", e.GetMsgA());
        return false;
    } catch(...) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "Unable to create dots shader");
        return false;
    }

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
    this->dotsShader.Release();
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
    GLint attrloc = ::glGetAttribLocationARB(this->pathlineShader.ProgramHandle(), "params");
    ::glEnableClientState(GL_VERTEX_ARRAY);
    ::glEnableVertexAttribArrayARB(attrloc);
    this->pathlineShader.SetParameter("paramSpan",
        spdc->MinTime(),
        1.0f / vislib::math::Max(1.0f, spdc->MaxTime() - spdc->MinTime()),
        spdc->MinSpeed(),
        1.0f / vislib::math::Max(1.0f, spdc->MaxSpeed() - spdc->MinSpeed()));

    const SolPathDataCall::Pathline *path = spdc->Pathlines();
    for (unsigned int p = 0; p < spdc->Count(); p++, path++) {
        ::glVertexPointer(4, GL_FLOAT, sizeof(SolPathDataCall::Vertex), path->data);
        ::glVertexAttribPointerARB(attrloc, 2, GL_FLOAT, GL_FALSE, sizeof(SolPathDataCall::Vertex), &path->data->time);
        ::glDrawArrays(GL_LINE_STRIP, 0, path->length);
    }
    this->pathlineShader.Disable();

    //this->dotsShader.Enable();
    //::glDisableVertexAttribArrayARB(attrloc);
    //attrloc = ::glGetAttribLocationARB(this->dotsShader.ProgramHandle(), "params");
    //::glEnableVertexAttribArrayARB(attrloc);
    //this->dotsShader.SetParameter("paramSpan",
    //    spdc->MinTime(),
    //    1.0f / vislib::math::Max(1.0f, spdc->MaxTime() - spdc->MinTime()),
    //    spdc->MinSpeed(),
    //    1.0f / vislib::math::Max(1.0f, spdc->MaxSpeed() - spdc->MinSpeed()));
    //::glPointSize(2.0f);
    //::glEnable(GL_POINT_SMOOTH);
    //::glEnable(GL_BLEND);
    //::glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    //::glColor3ub(255, 0, 0);
    //path = spdc->Pathlines();
    //for (unsigned int p = 0; p < spdc->Count(); p++, path++) {
    //    ::glVertexPointer(4, GL_FLOAT, sizeof(SolPathDataCall::Vertex), path->data);
    //    ::glVertexAttribPointerARB(attrloc, 2, GL_FLOAT, GL_FALSE, sizeof(SolPathDataCall::Vertex), &path->data->time);
    //    ::glDrawArrays(GL_POINTS, 0, path->length);
    //}
    //::glDisable(GL_POINT_SMOOTH);
    //this->dotsShader.Disable();

    ::glDisableVertexAttribArrayARB(attrloc);
    ::glDisableClientState(GL_VERTEX_ARRAY);

    return false;
}
