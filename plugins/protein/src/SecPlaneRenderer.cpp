//
// SecPlaneRenderer.cpp
//
// Copyright (C) 2013 by University of Stuttgart (VISUS).
// All rights reserved.
//
// Created on: Sep 13, 2013
//     Author: scharnkn
//

#include "stdafx.h"

#include "SecPlaneRenderer.h"
#include "protein_calls/VTIDataCall.h"
#include "ogl_error_check.h"
//#include "vislib_vector_typedefs.h"

#include "mmcore/view/AbstractCallRender3D.h"
#include "mmcore/param/FloatParam.h"
#include "mmcore/param/BoolParam.h"
#include "mmcore/param/EnumParam.h"
#include "mmcore/CoreInstance.h"

#include "vislib/sys/Log.h"

using namespace megamol;
using namespace megamol::core;
using namespace megamol::protein;


/*
 * SecPlaneRenderer::SecPlaneRenderer
 */
SecPlaneRenderer::SecPlaneRenderer(void) : view::Renderer3DModule(),
    textureSlot("getData", "Connects the slice rendering with data storage" ),
    shadingSlot("shading", "Determines the shading mode"),
    shadingMinTexSlot("min", "The minimum texture value (used for shading)"),
    shadingMaxTexSlot("max", "The maximum texture value (used for shading)"),
    licContrastSlot("licContrast", "LIC contrast"),
    licBrightnessSlot("licBrightness", "LIC licBrightness"),
    licDirSclSlot("licDirScl", "LIC stepsize scale factor"),
    licTCSclSlot("licTCScl", "LIC random noise texture coordinates scale"),
    isoValueSlot("isoValue", "Isovalue for isolines"),
    isoThreshSlot("isoThresh", "Threshold for isolines"),
    isoDistributionSlot("isoDistribution", "Determines the amount of isolines"),
    xPlaneSlot("xPlanePos", "Change the position of the x-Plane"),
    yPlaneSlot("yPlanePos", "Change the position of the y-Plane"),
    zPlaneSlot("zPlanePos", "Change the position of the z-Plane"),
    toggleXPlaneSlot("showXPlane", "Change the position of the x-Plane"),
    toggleYPlaneSlot("showYPlane", "Change the position of the y-Plane"),
    toggleZPlaneSlot("showZPlane", "Change the position of the z-Plane") {

    // Make texture slot available
	this->textureSlot.SetCompatibleCall<protein_calls::VTIDataCallDescription>();
    this->MakeSlotAvailable(&this->textureSlot);


    /* Init parameter slots */

    // Shading modes for slices
    param::EnumParam *srm = new core::param::EnumParam(0);
    srm->SetTypePair(0, "Density Map");
    srm->SetTypePair(1, "Potential Map");
    srm->SetTypePair(2, "LIC");
    srm->SetTypePair(3, "Isolines");
    this->shadingSlot << srm;
    this->MakeSlotAvailable(&this->shadingSlot);

    // Minimum texture value
    this->shadingMinTexSlot.SetParameter(new core::param::FloatParam(-1.0f));
    this->MakeSlotAvailable(&this->shadingMinTexSlot);

    // Maximum texture value
    this->shadingMaxTexSlot.SetParameter(new core::param::FloatParam(1.0f));
    this->MakeSlotAvailable(&this->shadingMaxTexSlot);

    // LIC contrast
    this->licContrastSlot.SetParameter(new core::param::FloatParam(1.0f, 0.0f));
    this->MakeSlotAvailable(&this->licContrastSlot);

    // LIC brightness
    this->licBrightnessSlot.SetParameter(new core::param::FloatParam(1.0f, 0.0f));
    this->MakeSlotAvailable(&this->licBrightnessSlot);

    // LIC stepsize scale
    this->licDirSclSlot.SetParameter(new core::param::FloatParam(1.0f, 0.0f));
    this->MakeSlotAvailable(&this->licDirSclSlot);

    // LIC tex coord scale
    this->licTCSclSlot.SetParameter(new core::param::FloatParam(1.0f, 0.0f));
    this->MakeSlotAvailable(&this->licTCSclSlot);

    // Isovalue
    this->isoValueSlot.SetParameter(new core::param::FloatParam(1.0f));
    this->MakeSlotAvailable(&this->isoValueSlot);

    // Threshold for rendering of isolines (determines thickness)
    this->isoThreshSlot.SetParameter(new core::param::FloatParam(1.0f, 0.0f));
    this->MakeSlotAvailable(&this->isoThreshSlot);

    // Determines the amount of isolines
    this->isoDistributionSlot.SetParameter(new core::param::FloatParam(1.0f, 0.0f));
    this->MakeSlotAvailable(&this->isoDistributionSlot);

    // X-plane position
    this->xPlaneSlot.SetParameter(new core::param::FloatParam(0.0f));
    this->MakeSlotAvailable(&this->xPlaneSlot);

    // Y-plane position
    this->yPlaneSlot.SetParameter(new core::param::FloatParam(0.0f));
    this->MakeSlotAvailable(&this->yPlaneSlot);

    // Z-plane position
    this->zPlaneSlot.SetParameter(new core::param::FloatParam(0.0f));
    this->MakeSlotAvailable(&this->zPlaneSlot);

    // X-plane visibility
    this->toggleXPlaneSlot.SetParameter(new core::param::BoolParam(false));
    this->MakeSlotAvailable(&this->toggleXPlaneSlot);

    // Y-plane visibility
    this->toggleYPlaneSlot.SetParameter(new core::param::BoolParam(false));
    this->MakeSlotAvailable(&this->toggleYPlaneSlot);

    // Z-plane visibility
    this->toggleZPlaneSlot.SetParameter(new core::param::BoolParam(true));
    this->MakeSlotAvailable(&this->toggleZPlaneSlot);
}


/*
 * SecPlaneRenderer::~SecPlaneRenderer
 */
SecPlaneRenderer::~SecPlaneRenderer(void) {
    this->Release();
}


/*
 * SecPlaneRenderer::create
 */
bool SecPlaneRenderer::create(void) {
    using namespace vislib::sys;
    using namespace vislib::graphics::gl;

    // Init extensions
    if(!ogl_IsVersionGEQ(2,0) || !areExtsAvailable("GL_EXT_texture3D GL_EXT_framebuffer_object GL_ARB_multitexture GL_ARB_draw_buffers GL_ARB_vertex_buffer_object")) {
        return false;
    }

    if (!vislib::graphics::gl::GLSLShader::InitialiseExtensions()) {
        return false;
    }

    // Load shader sources
    ShaderSource vertSrc, fragSrc, geomSrc;

    core::CoreInstance *ci = this->GetCoreInstance();
    if(!ci) return false;

    // Load slice shader
    if (!ci->ShaderSourceFactory().MakeShaderSource("protein::slice::vertex", vertSrc)) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
                "%s: Unable to load vertex shader source: slice shader", this->ClassName());
        return false;
    }
    if (!ci->ShaderSourceFactory().MakeShaderSource("protein::slice::fragment", fragSrc)) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
                "%s: Unable to load fragment shader source:  slice shader", this->ClassName());
        return false;
    }
    try {
        if (!this->sliceShader.Create(vertSrc.Code(), vertSrc.Count(), fragSrc.Code(), fragSrc.Count()))
            throw vislib::Exception("Generic creation failure", __FILE__, __LINE__);
    }
    catch (vislib::Exception &e){
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
                "%s: Unable to create slice shader: %s\n", this->ClassName(), e.GetMsgA());
        return false;
    }

    // Create random noise texture for LIC

    srand((unsigned)time(0));  // Init random number generator
    if (!this->initLIC()) {
        return false;
    }

    return true;
}


/*
 * SecPlaneRenderer::GetExtents
 */
bool SecPlaneRenderer::GetExtents(megamol::core::Call& call) {
    core::view::CallRender3D *cr3d = dynamic_cast<core::view::CallRender3D *>(&call);
    if (cr3d == NULL) {
        return false;
    }

    // Get extent of texture
	protein_calls::VTIDataCall *vti = this->textureSlot.CallAs<protein_calls::VTIDataCall>();
    if (vti == NULL) {
        return false;
    }
	if (!(*vti)(protein_calls::VTIDataCall::CallForGetExtent)) {
        return false;
    }

    this->bbox = vti->AccessBoundingBoxes();
    float scale;
    if(!vislib::math::IsEqual(this->bbox.ObjectSpaceBBox().LongestEdge(), 0.0f) ) {
        scale = 2.0f / this->bbox.ObjectSpaceBBox().LongestEdge();
    } else {
        scale = 1.0f;
    }
    this->bbox.MakeScaledWorld(scale);

    cr3d->AccessBoundingBoxes() = this->bbox;
    cr3d->SetTimeFramesCount(vti->FrameCount()); // TODO USe combined frame count

    return true;
}


/*
 * SecPlaneRenderer::initLIC
 */
bool SecPlaneRenderer::initLIC() {
    using namespace vislib::sys;

    // Create randbuffer
    float *randBuff = new float[32*32*32];
    for (int i = 0; i < 32*32*32; ++i) {
        float randVal = (float)rand()/float(RAND_MAX);
        randBuff[i]= randVal;
    }

    // Setup random noise texture
    glEnable(GL_TEXTURE_3D);
    if (glIsTexture(this->randNoiseTex)) {
        glDeleteTextures(1, &this->randNoiseTex);
    }
    glGenTextures(1, &this->randNoiseTex);
    glBindTexture(GL_TEXTURE_3D, this->randNoiseTex);

    glTexImage3DEXT(GL_TEXTURE_3D,
            0,
            GL_ALPHA,
            32, // Create random buffer with 32x32x32
            32,
            32,
            0,
            GL_ALPHA,
            GL_FLOAT,
            randBuff);

    //glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    //glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_REPEAT);
    glBindTexture(GL_TEXTURE_3D, 0);
    glDisable(GL_TEXTURE_3D);

    return CheckForGLError();;
}


/*
 * SecPlaneRenderer::release
 */
void SecPlaneRenderer::release(void) {
    if (glIsTexture(this->tex)) {
        glDeleteTextures(1, &this->tex);
    }
    if (glIsTexture(this->randNoiseTex)) {
        glDeleteTextures(1, &this->randNoiseTex);
    }
}


/*
 * SecPlaneRenderer::Render
 */
bool SecPlaneRenderer::Render(megamol::core::Call& call) {
    using namespace vislib::sys;

    // Get render call
    core::view::CallRender3D *cr3d = dynamic_cast<core::view::CallRender3D *>(&call);
    if (cr3d == NULL) {
        return false;
    }

	protein_calls::VTIDataCall *vti = this->textureSlot.CallAs<protein_calls::VTIDataCall>();
    if (vti == NULL) {
        return false;
    }

    // Set time
    vti->SetCalltime(cr3d->Time());
    vti->SetFrameID(static_cast<int>(cr3d->Time()), true);

    // Get data for this frame
	if (!(*vti)(protein_calls::VTIDataCall::CallForGetData)) {
        return false;
    }

    if (vti->GetPiecePointArraySize(0, 0) == 0) {
        return true;
    }

//    // DEBUG print texture values
//    for (int i = 0; i < vti->GetPiecePointArraySize(0, 0); ++i) {
//        printf("%f\n", ((const float*)(vti->GetPointDataByIdx(0, 0)))[i]);
//    }
//    // END DEBUG

    /* Init texture */

    // Note: The first array of the first piece is used
    // TODO Do not init texture in every frame

    //  Setup texture
    glEnable(GL_TEXTURE_3D);
    if (!glIsTexture(this->tex)) {
        glGenTextures(1, &this->tex);
    }
    glBindTexture(GL_TEXTURE_3D, this->tex);
    glTexImage3DEXT(GL_TEXTURE_3D,
            0,
            GL_RGBA32F,
            vti->GetGridsize().X(),
            vti->GetGridsize().Y(),
            vti->GetGridsize().Z(),
            0,
            GL_ALPHA,
            GL_FLOAT,
            vti->GetPointDataByIdx(0, 0));
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP);
    CheckForGLError();

    if (vti->GetPointDataArrayNumberOfComponents(0, 0) == 1) { // Scalar texture
    } else {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
                "%s: Invalid texture format (needs to be scalar)\n",
                this->ClassName());
    }

    /* Render slices */

    // Compute scale factor and scale world
    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();

    float scale;
    if(!vislib::math::IsEqual(this->bbox.ObjectSpaceBBox().LongestEdge(), 0.0f) ) {
        scale = 2.0f / this->bbox.ObjectSpaceBBox().LongestEdge();
    } else {
        scale = 1.0f;
    }
    glScalef(scale, scale, scale);

    glEnable(GL_DEPTH_TEST);
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    glDisable(GL_CULL_FACE);
    glDisable(GL_LIGHTING);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    Vec3f gridMinCoord = vti->GetOrigin();
    Vec3f gridMaxCoord;
    gridMaxCoord.SetX(vti->GetOrigin().X() + vti->GetSpacing().X() * (vti->GetGridsize().X()-1));
    gridMaxCoord.SetY(vti->GetOrigin().Y() + vti->GetSpacing().Y() * (vti->GetGridsize().Y()-1));
    gridMaxCoord.SetZ(vti->GetOrigin().Z() + vti->GetSpacing().Z() * (vti->GetGridsize().Z()-1));

    // Calc ws positions and tex coords for planes
    float xPlane = this->xPlaneSlot.Param<core::param::FloatParam>()->Value();
    float yPlane = this->yPlaneSlot.Param<core::param::FloatParam>()->Value();
    float zPlane = this->zPlaneSlot.Param<core::param::FloatParam>()->Value();
    float texCoordX = (xPlane - gridMinCoord[0])/(gridMaxCoord[0] - gridMinCoord[0]);
    float texCoordY = (yPlane - gridMinCoord[1])/(gridMaxCoord[1] - gridMinCoord[1]);
    float texCoordZ = (zPlane - gridMinCoord[2])/(gridMaxCoord[2] - gridMinCoord[2]);

    this->sliceShader.Enable();
    glUniform1iARB(this->sliceShader.ParameterLocation("tex"), 0);
    glUniform1iARB(this->sliceShader.ParameterLocation("randNoiseTex"), 1);
    glUniform1fARB(this->sliceShader.ParameterLocation("minTex"),
            this->shadingMinTexSlot.Param<core::param::FloatParam>()->Value());
    glUniform1fARB(this->sliceShader.ParameterLocation("maxTex"),
            this->shadingMaxTexSlot.Param<core::param::FloatParam>()->Value());
    glUniform1iARB(this->sliceShader.ParameterLocation("mode"),
            this->shadingSlot.Param<core::param::EnumParam>()->Value());
    glUniform1fARB(this->sliceShader.ParameterLocation("licContrast"),
            this->licContrastSlot.Param<core::param::FloatParam>()->Value());
    glUniform1fARB(this->sliceShader.ParameterLocation("licBrightness"),
            this->licBrightnessSlot.Param<core::param::FloatParam>()->Value());
    glUniform1fARB(this->sliceShader.ParameterLocation("licDirScl"),
            this->licDirSclSlot.Param<core::param::FloatParam>()->Value());
    glUniform1fARB(this->sliceShader.ParameterLocation("licTCScl"),
            this->licTCSclSlot.Param<core::param::FloatParam>()->Value());
    glUniform1fARB(this->sliceShader.ParameterLocation("isoval"),
            this->isoValueSlot.Param<core::param::FloatParam>()->Value());
    glUniform1fARB(this->sliceShader.ParameterLocation("isoThresh"),
            this->isoThreshSlot.Param<core::param::FloatParam>()->Value());
    glUniform1fARB(this->sliceShader.ParameterLocation("isoDistribution"),
            this->isoDistributionSlot.Param<core::param::FloatParam>()->Value());

    glActiveTextureARB(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_3D, this->randNoiseTex);

    glActiveTextureARB(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_3D, this->tex);

    if (this->toggleXPlaneSlot.Param<core::param::BoolParam>()->Value()) { // Render x plane
        glBegin(GL_QUADS);
        glTexCoord3f(texCoordX, 1.0f, 0.0f);
        glVertex3f(xPlane, gridMaxCoord[1], gridMinCoord[2]);
        glTexCoord3f(texCoordX, 0.0f, 0.0f);
        glVertex3f(xPlane, gridMinCoord[1], gridMinCoord[2]);
        glTexCoord3f(texCoordX, 0.0f, 1.0f);
        glVertex3f(xPlane, gridMinCoord[1], gridMaxCoord[2]);
        glTexCoord3f(texCoordX, 1.0f, 1.0f);
        glVertex3f(xPlane, gridMaxCoord[1], gridMaxCoord[2]);
        glEnd();
    }
    if (this->toggleYPlaneSlot.Param<core::param::BoolParam>()->Value()) { // Render y plane
        glBegin(GL_QUADS);
        glTexCoord3f(0.0f, texCoordY, 1.0f);
        glVertex3f(gridMinCoord[0], yPlane, gridMaxCoord[2]);
        glTexCoord3f( 0.0f, texCoordY, 0.0f);
        glVertex3f(gridMinCoord[0], yPlane, gridMinCoord[2]);
        glTexCoord3f(1.0f, texCoordY, 0.0f);
        glVertex3f(gridMaxCoord[0], yPlane, gridMinCoord[2]);
        glTexCoord3f(1.0f, texCoordY, 1.0f);
        glVertex3f(gridMaxCoord[0], yPlane, gridMaxCoord[2]);
        glEnd();
    }
    if (this->toggleZPlaneSlot.Param<core::param::BoolParam>()->Value()) { // Render z plane
        glBegin(GL_QUADS);
        glTexCoord3f(0.0f, 1.0f, texCoordZ);
        glVertex3f(gridMinCoord[0], gridMaxCoord[1], zPlane);
        glTexCoord3f(0.0f, 0.0f, texCoordZ);
        glVertex3f(gridMinCoord[0], gridMinCoord[1], zPlane);
        glTexCoord3f(1.0f, 0.0f, texCoordZ);
        glVertex3f(gridMaxCoord[0], gridMinCoord[1], zPlane);
        glTexCoord3f(1.0f, 1.0f, texCoordZ);
        glVertex3f(gridMaxCoord[0], gridMaxCoord[1], zPlane);
        glEnd();
    }

    this->sliceShader.Disable();
    glBindTexture(GL_TEXTURE_3D, 0);
    glDisable(GL_TEXTURE_3D);
    glPopMatrix();

    return CheckForGLError();
}
