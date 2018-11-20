//
// SurfacePotentialRendererSlave.cpp
//
// Copyright (C) 2013 by University of Stuttgart (VISUS).
// All rights reserved.
//
// Created on: May 31, 2013
//     Author: scharnkn
//


#include "stdafx.h"
#include "SurfacePotentialRendererSlave.h"
#include "VBODataCall.h"
#include "ogl_error_check.h"
#include "mmcore/CoreInstance.h"
#include "mmcore/param/FloatParam.h"

using namespace megamol;
using namespace megamol::protein_cuda;


/*
 * SurfacePotentialRendererSlave::SurfacePotentialRendererSlave
 */
SurfacePotentialRendererSlave::SurfacePotentialRendererSlave(void) : Renderer3DModuleDS(),
        vboSlot("vboIn", "Caller slot to obtain vbo data and extent"),
        surfAlphaSclSlot("alphaScl", "Transparency scale factor") {

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

    using namespace vislib::sys;
    using namespace vislib::graphics::gl;

    // Init extensions
    if (!ogl_IsVersionGEQ(2,0) || !areExtsAvailable("\
            GL_EXT_texture3D \
            GL_EXT_framebuffer_object \
            GL_ARB_draw_buffers \
            GL_ARB_vertex_buffer_object")) {
        return false;
    }
    if (!vislib::graphics::gl::GLSLShader::InitialiseExtensions()) {
        return false;
    }

    // Load shader sources
    ShaderSource vertSrc, fragSrc, geomSrc;

    core::CoreInstance *ci = this->GetCoreInstance();
    if (!ci) {
        return false;
    }

    // Load shader for per pixel lighting of the surface
    if (!this->GetCoreInstance()->ShaderSourceFactory().MakeShaderSource("electrostatics::pplsurface::vertex", vertSrc)) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "%s: Unable to load vertex shader source for the ppl shader",
                this->ClassName());
        return false;
    }
    // Load ppl fragment shader
    if (!this->GetCoreInstance()->ShaderSourceFactory().MakeShaderSource("electrostatics::pplsurface::fragment", fragSrc)) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "%s: Unable to load vertex shader source for the ppl shader", this->ClassName());
        return false;
    }
    try {
        if (!this->pplSurfaceShader.Create(vertSrc.Code(), vertSrc.Count(), fragSrc.Code(), fragSrc.Count())) {
            throw vislib::Exception("Generic creation failure", __FILE__, __LINE__ );
        }
    }
    catch (vislib::Exception &e) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "%s: Unable to create the ppl shader: %s\n", this->ClassName(), e.GetMsgA());
        return false;
    }

    return true;
}


/*
 * SurfacePotentialRendererSlave::GetExtents
 */
bool SurfacePotentialRendererSlave::GetExtents(core::Call& call) {

    core::view::CallRender3D *cr3d = dynamic_cast<core::view::CallRender3D *>(&call);
    if (cr3d == NULL) {
        return false;
    }

    VBODataCall *c = this->vboSlot.CallAs<VBODataCall>();
    if (c == NULL) {
        return false;
    }

    if (!(*c)(VBODataCall::CallForGetExtent)) {
        return false;
    }

    // Note: WS bbox hass already been scaled in master renderer so it does not
    // need to be scaled again
    this->bbox = c->GetBBox();
    cr3d->AccessBoundingBoxes() = this->bbox;
    cr3d->SetTimeFramesCount(c->GetFrameCnt());

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
    this->pplSurfaceShader.Release();
}


/*
 * SurfacePotentialRendererSlave::Render
 */
bool SurfacePotentialRendererSlave::Render(core::Call& call) {

    VBODataCall *c = this->vboSlot.CallAs<VBODataCall>();
    if (c == NULL) {
        return false;
    }

    if (!(*c)(VBODataCall::CallForGetData)) {
        return false;
    }

    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();

    // Apply scaling based on combined bounding box
    float scale;
    if (!vislib::math::IsEqual(this->bbox.ObjectSpaceBBox().LongestEdge(), 0.0f)) {
        scale = 2.0f/this->bbox.ObjectSpaceBBox().LongestEdge();
    } else {
        scale = 1.0f;
    }
    //printf("scale slave %f\n", scale);
    glScalef(scale, scale, scale);

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
bool SurfacePotentialRendererSlave::renderSurface(VBODataCall *c) {

    GLint attribLocPos, attribLocNormal, attribLocTexCoord;


    /* Get vertex attributes from vbo */

    glBindBufferARB(GL_ARRAY_BUFFER, c->GetVbo());
    CheckForGLError(); // OpenGL error check

    this->pplSurfaceShader.Enable();
    CheckForGLError(); // OpenGL error check

    // Note: glGetAttribLocation returnes -1 if the attribute if not used in
    // the shader code, because in this case the attribute is optimized out by
    // the compiler
    attribLocPos = glGetAttribLocationARB(this->pplSurfaceShader.ProgramHandle(), "pos");
    attribLocNormal = glGetAttribLocationARB(this->pplSurfaceShader.ProgramHandle(), "normal");
    attribLocTexCoord = glGetAttribLocationARB(this->pplSurfaceShader.ProgramHandle(), "texCoord");
    CheckForGLError(); // OpenGL error check

    glEnableVertexAttribArrayARB(attribLocPos);
    glEnableVertexAttribArrayARB(attribLocNormal);
    glEnableVertexAttribArrayARB(attribLocTexCoord);
    CheckForGLError(); // OpenGL error check

    glVertexAttribPointerARB(attribLocPos, 3, GL_FLOAT, GL_FALSE,
            c->GetDataStride()*sizeof(float),
            reinterpret_cast<void*>(c->GetDataOffsPosition()*sizeof(float)));
    glVertexAttribPointerARB(attribLocNormal, 3, GL_FLOAT, GL_FALSE,
            c->GetDataStride()*sizeof(float),
            reinterpret_cast<void*>(c->GetDataOffsNormal()*sizeof(float)));
    glVertexAttribPointerARB(attribLocTexCoord, 3, GL_FLOAT, GL_FALSE,
            c->GetDataStride()*sizeof(float),
            reinterpret_cast<void*>(c->GetDataOffsTexCoord()*sizeof(float)));
    CheckForGLError(); // OpenGL error check


    /* Render */

    // Set uniform vars
    glUniform1iARB(this->pplSurfaceShader.ParameterLocation("potentialTex"), 0);
    glUniform1iARB(this->pplSurfaceShader.ParameterLocation("colorMode"), 3); // Set color mode to 'surface potential'
    glUniform1iARB(this->pplSurfaceShader.ParameterLocation("renderMode"), 3); // Set render mode to 'fill'
    glUniform3fARB(this->pplSurfaceShader.ParameterLocation("colorMin"), 0.75f, 0.01f, 0.15f);
    glUniform3fARB(this->pplSurfaceShader.ParameterLocation("colorZero"), 1.0f, 1.0f, 1.0f);
    glUniform3fARB(this->pplSurfaceShader.ParameterLocation("colorMax"), 0.23f, 0.29f, 0.75f);
    glUniform3fARB(this->pplSurfaceShader.ParameterLocation("colorUniform"), 1.0, 0.0, 0.0);
    glUniform1fARB(this->pplSurfaceShader.ParameterLocation("minPotential"), c->GetTexValMin());
    glUniform1fARB(this->pplSurfaceShader.ParameterLocation("midPotential"), 0.0f);
    glUniform1fARB(this->pplSurfaceShader.ParameterLocation("maxPotential"), c->GetTexValMax());
    glUniform1fARB(this->pplSurfaceShader.ParameterLocation("alphaScl"),
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

    glDrawElements(GL_TRIANGLES,
            c->GetTriangleCnt()*3,
            GL_UNSIGNED_INT,
            reinterpret_cast<void*>(0));

//    glDrawArrays(GL_POINTS, 0, 3*vertexCnt); // DEBUG

    this->pplSurfaceShader.Disable();

    glDisableVertexAttribArrayARB(attribLocPos);
    glDisableVertexAttribArrayARB(attribLocNormal);
    glDisableVertexAttribArrayARB(attribLocTexCoord);
    CheckForGLError(); // OpenGL error check

    // Switch back to normal pointer operation by binding with 0
    glBindBufferARB(GL_ELEMENT_ARRAY_BUFFER_ARB, 0);
    glBindBufferARB(GL_ARRAY_BUFFER_ARB, 0);

    return CheckForGLError(); // OpenGL error check
}




