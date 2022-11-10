/*
 * SolPathRenderer.cpp
 *
 * Copyright (C) 2010 by VISUS (University of Stuttgart)
 * Alle Rechte vorbehalten.
 */

#include "SolPathRenderer.h"
#include "mmcore/CoreInstance.h"
#include "mmcore/factories/CallAutoDescription.h"
#include "mmcore_gl/utility/ShaderFactory.h"
#include "mmstd/renderer/CallRender3D.h"
#include "protein/SolPathDataCall.h"
#include "vislib/math/mathfunctions.h"
#include "vislib_gl/graphics/gl/IncludeAllGL.h"

using namespace megamol;
using namespace megamol::protein_gl;


/*
 * SolPathRenderer::SolPathRenderer
 */
SolPathRenderer::SolPathRenderer(void)
        : mmstd_gl::Renderer3DModuleGL()
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

    auto const shader_options = msf::ShaderFactoryOptionsOpenGL(GetCoreInstance()->GetShaderPaths());

    try {
        pathlineShader = core::utility::make_glowl_shader("pathlineShader", shader_options,
            "protein_gl/solpath/pathline.vert.glsl", "protein_gl/solpath/pathline.frag.glsl");

        dotsShader = core::utility::make_glowl_shader(
            "dotsShader", shader_options, "protein_gl/solpath/dots.vert.glsl", "protein_gl/solpath/dots.vert.glsl");

    } catch (std::exception& e) {
        Log::DefaultLog.WriteError(("SolPathRenderer: " + std::string(e.what())).c_str());
        return false;
    }

    return true;
}

/*
 * SolPathRenderer::GetExtents
 */
bool SolPathRenderer::GetExtents(mmstd_gl::CallRender3DGL& call) {
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
void SolPathRenderer::release(void) {}


/*
 * SolPathRenderer::Render
 */
bool SolPathRenderer::Render(mmstd_gl::CallRender3DGL& call) {
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

    this->pathlineShader->use();
    GLint attrloc = ::glGetAttribLocationARB(this->pathlineShader->getHandle(), "params");
    ::glEnableClientState(GL_VERTEX_ARRAY);
    ::glEnableVertexAttribArrayARB(attrloc);
    this->pathlineShader->setUniform("paramSpan", spdc->MinTime(),
        1.0f / vislib::math::Max(1.0f, spdc->MaxTime() - spdc->MinTime()), spdc->MinSpeed(),
        1.0f / vislib::math::Max(1.0f, spdc->MaxSpeed() - spdc->MinSpeed()));

    const protein::SolPathDataCall::Pathline* path = spdc->Pathlines();
    for (unsigned int p = 0; p < spdc->Count(); p++, path++) {
        ::glVertexPointer(4, GL_FLOAT, sizeof(protein::SolPathDataCall::Vertex), path->data);
        ::glVertexAttribPointerARB(
            attrloc, 2, GL_FLOAT, GL_FALSE, sizeof(protein::SolPathDataCall::Vertex), &path->data->time);
        ::glDrawArrays(GL_LINE_STRIP, 0, path->length);
    }
    glUseProgram(0);

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
