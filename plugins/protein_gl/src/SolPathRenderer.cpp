/*
 * SolPathRenderer.cpp
 *
 * Copyright (C) 2010 by VISUS (University of Stuttgart)
 * Alle Rechte vorbehalten.
 */

#include "SolPathRenderer.h"
#include "mmcore/CoreInstance.h"
#include "mmcore/factories/CallAutoDescription.h"
#include "mmcore/view/CallRender3D.h"
#include "mmcore_gl/utility/ShaderSourceFactory.h"
#include "protein/SolPathDataCall.h"
#include "vislib/math/mathfunctions.h"
#include "vislib_gl/graphics/gl/IncludeAllGL.h"
#include "vislib_gl/graphics/gl/ShaderSource.h"

using namespace megamol;
using namespace megamol::protein_gl;


/*
 * SolPathRenderer::SolPathRenderer
 */
SolPathRenderer::SolPathRenderer(void)
        : core_gl::view::Renderer3DModuleGL()
        , getdataslot("getdata", "Fetches data")
        , pathlineShader() {

    this->getdataslot.SetCompatibleCall<core::factories::CallAutoDescription<protein::SolPathDataCall>>();
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
    using megamol::core::utility::log::Log;

    auto ssf = std::make_shared<core_gl::utility::ShaderSourceFactory>(instance()->Configuration().ShaderDirectories());

    try {
        vislib_gl::graphics::gl::ShaderSource vertSrc;
        vislib_gl::graphics::gl::ShaderSource fragSrc;

        if (!ssf->MakeShaderSource("solpath::pathline::vert", vertSrc)) {
            Log::DefaultLog.WriteError("Unable to load pathline vertex shader source");
            return false;
        }
        if (!ssf->MakeShaderSource("solpath::pathline::frag", fragSrc)) {
            Log::DefaultLog.WriteError("Unable to load pathline fragment shader source");
            return false;
        }

        if (!this->pathlineShader.Create(vertSrc.Code(), vertSrc.Count(), fragSrc.Code(), fragSrc.Count())) {
            Log::DefaultLog.WriteError("Unable to create pathline shader");
            return false;
        }

    } catch (vislib::Exception e) {
        Log::DefaultLog.WriteError("Unable to create pathline shader: %s", e.GetMsgA());
        return false;
    } catch (...) {
        Log::DefaultLog.WriteError("Unable to create pathline shader");
        return false;
    }

    try {
        vislib_gl::graphics::gl::ShaderSource vertSrc;
        vislib_gl::graphics::gl::ShaderSource fragSrc;

        if (!ssf->MakeShaderSource("solpath::dots::vert", vertSrc)) {
            Log::DefaultLog.WriteError("Unable to load dots vertex shader source");
            return false;
        }
        if (!ssf->MakeShaderSource("solpath::dots::frag", fragSrc)) {
            Log::DefaultLog.WriteError("Unable to load dots fragment shader source");
            return false;
        }

        if (!this->dotsShader.Create(vertSrc.Code(), vertSrc.Count(), fragSrc.Code(), fragSrc.Count())) {
            Log::DefaultLog.WriteError("Unable to create dots shader");
            return false;
        }

    } catch (vislib::Exception e) {
        Log::DefaultLog.WriteError("Unable to create dots shader: %s", e.GetMsgA());
        return false;
    } catch (...) {
        Log::DefaultLog.WriteError("Unable to create dots shader");
        return false;
    }

    return true;
}

/*
 * SolPathRenderer::GetExtents
 */
bool SolPathRenderer::GetExtents(core_gl::view::CallRender3DGL& call) {
    core::view::CallRender3D* cr3d = dynamic_cast<core::view::CallRender3D*>(&call);
    if (cr3d == NULL)
        return false;

    protein::SolPathDataCall* spdc = this->getdataslot.CallAs<protein::SolPathDataCall>();
    if (spdc == NULL)
        return false;

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
bool SolPathRenderer::Render(core_gl::view::CallRender3DGL& call) {
    core::view::CallRender3D* cr3d = dynamic_cast<core::view::CallRender3D*>(&call);
    if (cr3d == NULL)
        return false;

    protein::SolPathDataCall* spdc = this->getdataslot.CallAs<protein::SolPathDataCall>();
    if (spdc == NULL)
        return false;

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
    this->pathlineShader.SetParameter("paramSpan", spdc->MinTime(),
        1.0f / vislib::math::Max(1.0f, spdc->MaxTime() - spdc->MinTime()), spdc->MinSpeed(),
        1.0f / vislib::math::Max(1.0f, spdc->MaxSpeed() - spdc->MinSpeed()));

    const protein::SolPathDataCall::Pathline* path = spdc->Pathlines();
    for (unsigned int p = 0; p < spdc->Count(); p++, path++) {
        ::glVertexPointer(4, GL_FLOAT, sizeof(protein::SolPathDataCall::Vertex), path->data);
        ::glVertexAttribPointerARB(
            attrloc, 2, GL_FLOAT, GL_FALSE, sizeof(protein::SolPathDataCall::Vertex), &path->data->time);
        ::glDrawArrays(GL_LINE_STRIP, 0, path->length);
    }
    this->pathlineShader.Disable();

    // this->dotsShader.Enable();
    //::glDisableVertexAttribArrayARB(attrloc);
    // attrloc = ::glGetAttribLocationARB(this->dotsShader.ProgramHandle(), "params");
    //::glEnableVertexAttribArrayARB(attrloc);
    // this->dotsShader.SetParameter("paramSpan",
    //    spdc->MinTime(),
    //    1.0f / vislib::math::Max(1.0f, spdc->MaxTime() - spdc->MinTime()),
    //    spdc->MinSpeed(),
    //    1.0f / vislib::math::Max(1.0f, spdc->MaxSpeed() - spdc->MinSpeed()));
    //::glPointSize(2.0f);
    //::glEnable(GL_POINT_SMOOTH);
    //::glEnable(GL_BLEND);
    //::glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    //::glColor3ub(255, 0, 0);
    // path = spdc->Pathlines();
    // for (unsigned int p = 0; p < spdc->Count(); p++, path++) {
    //    ::glVertexPointer(4, GL_FLOAT, sizeof(SolPathDataCall::Vertex), path->data);
    //    ::glVertexAttribPointerARB(attrloc, 2, GL_FLOAT, GL_FALSE, sizeof(SolPathDataCall::Vertex),
    //    &path->data->time);
    //    ::glDrawArrays(GL_POINTS, 0, path->length);
    //}
    //::glDisable(GL_POINT_SMOOTH);
    // this->dotsShader.Disable();

    ::glDisableVertexAttribArrayARB(attrloc);
    ::glDisableClientState(GL_VERTEX_ARRAY);

    return false;
}
