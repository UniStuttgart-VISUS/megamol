//
// SurfacePotentialRendererSlave.cpp
//
// Copyright (C) 2013 by University of Stuttgart (VISUS).
// All rights reserved.
//
// Created on: May 31, 2013
//     Author: scharnkn
//


#include "SurfacePotentialRendererSlave.h"
#include "VBODataCall.h"
#include "mmcore/CoreInstance.h"
#include "mmcore/param/FloatParam.h"
#include "mmcore_gl/utility/ShaderFactory.h"
#include "ogl_error_check.h"

using namespace megamol;
using namespace megamol::protein_cuda;
using namespace megamol::core::utility::log;


/*
 * SurfacePotentialRendererSlave::SurfacePotentialRendererSlave
 */
SurfacePotentialRendererSlave::SurfacePotentialRendererSlave(void)
        : Renderer3DModuleGL()
        , vboSlot("vboIn", "Caller slot to obtain vbo data and extent")
        , surfAlphaSclSlot("alphaScl", "Transparency scale factor") {

    // Data caller for vbo
    this->vboSlot.SetCompatibleCall<VBODataCallDescription>();
    this->MakeSlotAvailable(&this->vboSlot);

    // Param for transparency scaling
    this->surfAlphaSclSlot.SetParameter(new core::param::FloatParam(1.0f, 0.0f, 1.0f));
    this->MakeSlotAvailable(&this->surfAlphaSclSlot);
}


/*
 * SurfacePotentialRendererSlave::~SurfacePotentialRendererSlave
 */
SurfacePotentialRendererSlave::~SurfacePotentialRendererSlave(void) {
    this->Release();
}


/*
 * SurfacePotentialRendererSlave::create
 */
bool SurfacePotentialRendererSlave::create(void) {

    using namespace vislib_gl::graphics::gl;

    // Init extensions
    /*if (!ogl_IsVersionGEQ(2, 0) || !areExtsAvailable("\
            GL_EXT_texture3D \
            GL_EXT_framebuffer_object \
            GL_ARB_draw_buffers \
            GL_ARB_vertex_buffer_object")) {
        return false;
    }*/

    // Load shader sources
    auto const shader_options = msf::ShaderFactoryOptionsOpenGL(GetCoreInstance()->GetShaderPaths());

    try {
        this->pplSurfaceShader = core::utility::make_glowl_shader("pplSurfaceShader", shader_options,
            "protein_cuda/electrostatics/pplsurface.vert.glsl", "protein_cuda/electrostatics/pplsurface.frag.glsl");

    } catch (std::exception& e) {
        Log::DefaultLog.WriteError(("SurfacePotentialRendererSlave: " + std::string(e.what())).c_str());
        return false;
    }

    return true;
}


/*
 * SurfacePotentialRendererSlave::GetExtents
 */
bool SurfacePotentialRendererSlave::GetExtents(mmstd_gl::CallRender3DGL& call) {
    VBODataCall* c = this->vboSlot.CallAs<VBODataCall>();
    if (c == NULL) {
        return false;
    }

    if (!(*c)(VBODataCall::CallForGetExtent)) {
        return false;
    }

    // Note: WS bbox hass already been scaled in master renderer so it does not
    // need to be scaled again
    this->bbox = c->GetBBox();
    call.AccessBoundingBoxes() = this->bbox;
    call.SetTimeFramesCount(c->GetFrameCnt());

    //    printf("Slave Call3d Object Space BBOX %f %f %f, %f %f %f\n",
    //            cr3d->AccessBoundingBoxes().ObjectSpaceBBox().Left(),
    //            cr3d->AccessBoundingBoxes().ObjectSpaceBBox().Bottom(),
    //            cr3d->AccessBoundingBoxes().ObjectSpaceBBox().Back(),
    //            cr3d->AccessBoundingBoxes().ObjectSpaceBBox().Right(),
    //            cr3d->AccessBoundingBoxes().ObjectSpaceBBox().Top(),
    //            cr3d->AccessBoundingBoxes().ObjectSpaceBBox().Front());
    //
    //    printf("Slave Call3d World Space  BBOX %f %f %f, %f %f %f\n",
    //            cr3d->AccessBoundingBoxes().WorldSpaceBBox().Left(),
    //            cr3d->AccessBoundingBoxes().WorldSpaceBBox().Bottom(),
    //            cr3d->AccessBoundingBoxes().WorldSpaceBBox().Back(),
    //            cr3d->AccessBoundingBoxes().WorldSpaceBBox().Right(),
    //            cr3d->AccessBoundingBoxes().WorldSpaceBBox().Top(),
    //            cr3d->AccessBoundingBoxes().WorldSpaceBBox().Front());
    //
    //    printf("Slave Call3d object Space clip BBOX %f %f %f, %f %f %f\n",
    //            cr3d->AccessBoundingBoxes().ObjectSpaceClipBox().Left(),
    //            cr3d->AccessBoundingBoxes().ObjectSpaceClipBox().Bottom(),
    //            cr3d->AccessBoundingBoxes().ObjectSpaceClipBox().Back(),
    //            cr3d->AccessBoundingBoxes().ObjectSpaceClipBox().Right(),
    //            cr3d->AccessBoundingBoxes().ObjectSpaceClipBox().Top(),
    //            cr3d->AccessBoundingBoxes().ObjectSpaceClipBox().Front());
    //
    //    printf("Slave Call3d World Space clip BBOX %f %f %f, %f %f %f\n",
    //            cr3d->AccessBoundingBoxes().WorldSpaceClipBox().Left(),
    //            cr3d->AccessBoundingBoxes().WorldSpaceClipBox().Bottom(),
    //            cr3d->AccessBoundingBoxes().WorldSpaceClipBox().Back(),
    //            cr3d->AccessBoundingBoxes().WorldSpaceClipBox().Right(),
    //            cr3d->AccessBoundingBoxes().WorldSpaceClipBox().Top(),
    //            cr3d->AccessBoundingBoxes().WorldSpaceClipBox().Front());

    return true;
}


/*
 * SurfacePotentialRendererSlave::release
 */
void SurfacePotentialRendererSlave::release(void) {
    this->pplSurfaceShader.reset();
}


/*
 * SurfacePotentialRendererSlave::Render
 */
bool SurfacePotentialRendererSlave::Render(mmstd_gl::CallRender3DGL& call) {

    VBODataCall* c = this->vboSlot.CallAs<VBODataCall>();
    if (c == NULL) {
        return false;
    }

    if (!(*c)(VBODataCall::CallForGetData)) {
        return false;
    }

    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();

    //    // DEBUG Print modelview matrix
    //    GLfloat matrix[16];
    //    printf("Modelview matrix slave\n");
    //    glGetFloatv (GL_MODELVIEW_MATRIX, matrix);
    //    for (int i = 0; i < 4; ++i) {
    //       for (int j = 0; j < 4; ++j)  {
    //           printf("%.4f ", matrix[j*4+i]);
    //       }
    //       printf("\n");
    //    }
    //    // END DEBUG


    if (!this->renderSurface(c)) {
        return false;
    }

    glPopMatrix();

    return CheckForGLError();
}


/*
 * SurfacePotentialRendererSlave::renderSurface
 */
bool SurfacePotentialRendererSlave::renderSurface(VBODataCall* c) {

    GLint attribLocPos, attribLocNormal, attribLocTexCoord;


    /* Get vertex attributes from vbo */

    glBindBufferARB(GL_ARRAY_BUFFER, c->GetVbo());
    CheckForGLError(); // OpenGL error check

    this->pplSurfaceShader->use();
    CheckForGLError(); // OpenGL error check

    // Note: glGetAttribLocation returnes -1 if the attribute if not used in
    // the shader code, because in this case the attribute is optimized out by
    // the compiler
    attribLocPos = glGetAttribLocationARB(this->pplSurfaceShader->getHandle(), "pos");
    attribLocNormal = glGetAttribLocationARB(this->pplSurfaceShader->getHandle(), "normal");
    attribLocTexCoord = glGetAttribLocationARB(this->pplSurfaceShader->getHandle(), "texCoord");
    CheckForGLError(); // OpenGL error check

    glEnableVertexAttribArrayARB(attribLocPos);
    glEnableVertexAttribArrayARB(attribLocNormal);
    glEnableVertexAttribArrayARB(attribLocTexCoord);
    CheckForGLError(); // OpenGL error check

    glVertexAttribPointerARB(attribLocPos, 3, GL_FLOAT, GL_FALSE, c->GetDataStride() * sizeof(float),
        reinterpret_cast<void*>(c->GetDataOffsPosition() * sizeof(float)));
    glVertexAttribPointerARB(attribLocNormal, 3, GL_FLOAT, GL_FALSE, c->GetDataStride() * sizeof(float),
        reinterpret_cast<void*>(c->GetDataOffsNormal() * sizeof(float)));
    glVertexAttribPointerARB(attribLocTexCoord, 3, GL_FLOAT, GL_FALSE, c->GetDataStride() * sizeof(float),
        reinterpret_cast<void*>(c->GetDataOffsTexCoord() * sizeof(float)));
    CheckForGLError(); // OpenGL error check


    /* Render */

    // Set uniform vars
    glUniform1iARB(this->pplSurfaceShader->getUniformLocation("potentialTex"), 0);
    glUniform1iARB(this->pplSurfaceShader->getUniformLocation("colorMode"), 3); // Set color mode to 'surface potential'
    glUniform1iARB(this->pplSurfaceShader->getUniformLocation("renderMode"), 3); // Set render mode to 'fill'
    glUniform3fARB(this->pplSurfaceShader->getUniformLocation("colorMin"), 0.75f, 0.01f, 0.15f);
    glUniform3fARB(this->pplSurfaceShader->getUniformLocation("colorZero"), 1.0f, 1.0f, 1.0f);
    glUniform3fARB(this->pplSurfaceShader->getUniformLocation("colorMax"), 0.23f, 0.29f, 0.75f);
    glUniform3fARB(this->pplSurfaceShader->getUniformLocation("colorUniform"), 1.0, 0.0, 0.0);
    glUniform1fARB(this->pplSurfaceShader->getUniformLocation("minPotential"), c->GetTexValMin());
    glUniform1fARB(this->pplSurfaceShader->getUniformLocation("midPotential"), 0.0f);
    glUniform1fARB(this->pplSurfaceShader->getUniformLocation("maxPotential"), c->GetTexValMax());
    glUniform1fARB(this->pplSurfaceShader->getUniformLocation("alphaScl"),
        this->surfAlphaSclSlot.Param<core::param::FloatParam>()->Value());

    glActiveTextureARB(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_3D, c->GetTexId());
    CheckForGLError(); // OpenGL error check

    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glDisable(GL_CULL_FACE);
    glDisable(GL_LIGHTING);

    glBindBufferARB(GL_ELEMENT_ARRAY_BUFFER, c->GetVboTriangleIdx());
    CheckForGLError(); // OpenGL error check

    glDrawElements(GL_TRIANGLES, c->GetTriangleCnt() * 3, GL_UNSIGNED_INT, reinterpret_cast<void*>(0));

    //    glDrawArrays(GL_POINTS, 0, 3*vertexCnt); // DEBUG

    glUseProgram(0);

    glDisableVertexAttribArrayARB(attribLocPos);
    glDisableVertexAttribArrayARB(attribLocNormal);
    glDisableVertexAttribArrayARB(attribLocTexCoord);
    CheckForGLError(); // OpenGL error check

    // Switch back to normal pointer operation by binding with 0
    glBindBufferARB(GL_ELEMENT_ARRAY_BUFFER_ARB, 0);
    glBindBufferARB(GL_ARRAY_BUFFER_ARB, 0);

    return CheckForGLError(); // OpenGL error check
}
