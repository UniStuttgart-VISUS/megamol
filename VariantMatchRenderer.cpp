//
// VariantMatchRenderer.cpp
//
// Copyright (C) 2013 by University of Stuttgart (VISUS).
// All rights reserved.
//
// Created on: Jun 15, 2013
//     Author: scharnkn
//

#include <stdafx.h>
#include "VariantMatchRenderer.h"
#include "VariantMatchDataCall.h"
#include "ogl_error_check.h"

#include <param/FloatParam.h>
#include <CoreInstance.h>
#include <vislib/SimpleFont.h>
#include <vislib/OutlineFont.h>
#include <vislib/ShaderSource.h>
#include <vislib/Log.h>

using namespace megamol;
using namespace megamol::protein;


/*
 * VariantMatchRenderer::VariantMatchRenderer
 */
VariantMatchRenderer::VariantMatchRenderer(void) : Renderer2DModule () ,
        dataCallerSlot("getData", "Connects the rendering with data storage" ),
        labelSpaceSlot("labelSpace", "Fraction of the screen that is used to render labels" ),
        labelSizeSlot("labelSize", "Font size used to render labels" ), matrixTex(0) {

    // Data caller slot to get matching matrix
    this->dataCallerSlot.SetCompatibleCall<VariantMatchDataCallDescription>();
    this->MakeSlotAvailable(&this->dataCallerSlot);

    // Parameter for label space
    this->labelSpace = 0.25f;
    this->labelSpaceSlot.SetParameter(new core::param::FloatParam(this->labelSpace, 0.0f, 2.0f));
    this->MakeSlotAvailable(&this->labelSpaceSlot);

    // Parameter for label size
    this->labelSize = 0.1f;
    this->labelSizeSlot.SetParameter(new core::param::FloatParam(this->labelSize, 0.0f));
    this->MakeSlotAvailable(&this->labelSizeSlot);

}


/*
 * VariantMatchRenderer::~VariantMatchRenderer
 */
VariantMatchRenderer::~VariantMatchRenderer(void) {
    this->Release();
}


/*
 * VariantMatchRenderer::create
 */
bool VariantMatchRenderer::create(void) {
    vislib::graphics::gl::ShaderSource vertSrc;
    vislib::graphics::gl::ShaderSource fragSrc;

    megamol::core::CoreInstance *ci = this->GetCoreInstance();
    if(!ci) {
        return false;
    }

    if(!glh_init_extensions("GL_EXT_framebuffer_object GL_ARB_draw_buffers")) {
        return false;
    }
    if(!vislib::graphics::gl::GLSLShader::InitialiseExtensions()) {
        return false;
    }

    if (!glh_init_extensions("GL_ARB_texture_non_power_of_two")) {
        return false;
    }

    // Try to load the ssao shader
    if(!ci->ShaderSourceFactory().MakeShaderSource("2dplot::variantMatrix::vertex", vertSrc)) {
        vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
            "%s: Unable to load variant matrix vertex shader source", this->ClassName() );
        return false;
    }
    if(!ci->ShaderSourceFactory().MakeShaderSource("2dplot::variantMatrix::fragment", fragSrc)) {
        vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
            "%s: Unable to load variant matrix fragment shader source", this->ClassName() );
        return false;
    }
    try {
        if(!this->matrixTexShader.Create(vertSrc.Code(), vertSrc.Count(), fragSrc.Code(), fragSrc.Count()))
            throw vislib::Exception("Generic creation failure", __FILE__, __LINE__);
    }
    catch(vislib::Exception &e){
        vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
            "%s: Unable to create shader: %s\n", this->ClassName(), e.GetMsgA());
        return false;
    }

    return true;
}


/*
 * VariantMatchRenderer::GetExtents
 */
bool VariantMatchRenderer::GetExtents(megamol::core::view::CallRender2D& call) {
    return true;
}


/*
 * VariantMatchRenderer::release
 */
void VariantMatchRenderer::release(void) {
    this->matrixTexShader.Release();
    CheckForGLError();
    if (this->matrixTex) {
        glDeleteTextures(1, &this->matrixTex);
    }
    CheckForGLError();
}


/*
 * VariantMatchRenderer::Render
 */
bool VariantMatchRenderer::Render(megamol::core::view::CallRender2D& call) {

    float gridStep, gridHalfStep;

    // Update parameters
    this->updateParams();

    // Get pointer to VariantMatchDataCall
    VariantMatchDataCall *vmc = this->dataCallerSlot.CallAs<VariantMatchDataCall>();
    if(vmc == NULL) {
        return false;
    }

    //  Execute call for Getdata
    if (!(*vmc)(VariantMatchDataCall::CallForGetData)) {
        return false;
    }


    // Init matrix texture
    // TODO Only do this if necessary
    glEnable(GL_TEXTURE_2D);
    if (this->matrixTex) {
        ::glDeleteTextures(1, &this->matrixTex);
    }
    glGenTextures(1, &this->matrixTex);
    glBindTexture(GL_TEXTURE_2D, this->matrixTex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexImage2D(GL_TEXTURE_2D, 0,GL_RGBA32F,
            vmc->GetVariantCnt(),
            vmc->GetVariantCnt(),
            0, GL_ALPHA, GL_FLOAT,
            vmc->GetMatch());

//    // DEBUG Print matrix values
//    printf("Variant count %u\n", vmc->GetVariantCnt());
//    for (int i = 0; i < vmc->GetVariantCnt(); ++i) {
//        for (int j = 0; j < vmc->GetVariantCnt(); ++j) {
//            printf("%.2f ", vmc->GetMatch()[j*vmc->GetVariantCnt() + i]);
//        }
//        printf("\n");
//    }
//    // END DEBUG

    ::glMatrixMode(GL_PROJECTION);
    ::glPushMatrix();
    ::glLoadIdentity();

    ::glMatrixMode(GL_MODELVIEW);
    ::glPushMatrix();
    ::glLoadIdentity();

    gridStep = (2.0 - this->labelSpace)/static_cast<float>(vmc->GetVariantCnt());
    gridHalfStep = gridStep*0.5f;

//    printf("Min %f, max %f\n", vmc->GetMin(), vmc->GetMax());

    // Draw matrix texture
//    ::glColor3f(0.0f, 0.6f, 0.6f);
    this->matrixTexShader.Enable();
    glUniform1iARB(this->matrixTexShader.ParameterLocation("matrixTex"), 0);
    glUniform1fARB(this->matrixTexShader.ParameterLocation("minVal"), vmc->GetMin());
    glUniform1fARB(this->matrixTexShader.ParameterLocation("maxVal"), vmc->GetMax());
    ::glBegin(GL_QUADS);

        ::glTexCoord2f(1.0f, 0.0f);
        ::glVertex2f(-1.0 + this->labelSpace, -1.0);

        ::glTexCoord2f(1.0f, 1.0f);
        ::glVertex2f( 1.0, -1.0);

        ::glTexCoord2f(0.0f, 1.0f);
        ::glVertex2f( 1.0f,  1.0f - this->labelSpace);

        ::glTexCoord2f(0.0f, 0.0f);
        ::glVertex2f(-1.0 + this->labelSpace,  1.0f - this->labelSpace);

    ::glEnd();
    this->matrixTexShader.Disable();

    glBindTexture(GL_TEXTURE_2D, 0);
    glDisable(GL_TEXTURE_2D);

    ::glColor3f(1.0f, 1.0f, 1.0f);
    ::glLineWidth(3);
    // Draw vertical lines
    ::glBegin(GL_LINES);
        for (int i = 0; i < static_cast<int>(vmc->GetVariantCnt()); ++i) {
            ::glVertex2f((this->labelSpace+gridStep*i)-1.0, -1.0);
            ::glVertex2f((this->labelSpace+gridStep*i)-1.0,  1.0-this->labelSpace);
        }
    ::glEnd();

    // Draw horizontal lines
    ::glBegin(GL_LINES);
        for (int i = 1; i <= static_cast<int>(vmc->GetVariantCnt()); ++i) {
            ::glVertex2f(-1.0+this->labelSpace, gridStep*i-1.0);
            ::glVertex2f( 1.0, gridStep*i-1.0);
        }
    ::glEnd();


    glColor3f(0.0, 0.0, 0.0);
    vislib::graphics::gl::SimpleFont f;
    vislib::StringA str;
    if (!f.Initialise()) {
        return false;
    }
//    f.SetSize(0.5f);
    // Draw vertical labels
    for (int i = 1; i <= static_cast<int>(vmc->GetVariantCnt()); ++i) {
        str.Format("Variant %2i", static_cast<int>(vmc->GetVariantCnt()) - i);
        f.DrawString(-1.0 + this->labelSpace*0.5, gridStep*i-1.0 - gridHalfStep,
                this->labelSize,
                true,
                str.PeekBuffer(),
                vislib::graphics::AbstractFont::ALIGN_CENTER_MIDDLE);
    }
    // Draw horizontal labels
    ::glMatrixMode(GL_MODELVIEW);
    for (int i = 0; i < static_cast<int>(vmc->GetVariantCnt()); ++i) {
        ::glPushMatrix();
        ::glLoadIdentity();
        ::glTranslatef((this->labelSpace+gridStep*i)-1.0+ gridHalfStep, 1.0 - this->labelSpace*0.5, 0.0);
        ::glRotatef(80, 0.0, 0.0, 1.0);
        str.Format("Variant %2i", i);
        f.DrawString(
                //(this->labelSpace+gridStep*i)-1.0+ gridHalfStep,
                0.0, 0.0,
                //1.0 - gridHalfStep,
                this->labelSize,
                true,
                str.PeekBuffer(),
                vislib::graphics::AbstractFont::ALIGN_CENTER_MIDDLE);
        ::glPopMatrix();
    }

    ::glPopMatrix();

    ::glMatrixMode(GL_PROJECTION);
    ::glPopMatrix();


    return true;
}


/*
 * VariantMatchRenderer::updateParams
 */
void VariantMatchRenderer::updateParams() {

    // Label space
    if (this->labelSpaceSlot.IsDirty()) {
        this->labelSpace = this->labelSpaceSlot.Param<core::param::FloatParam>()->Value();
        this->labelSpaceSlot.ResetDirty();
    }

    // Label size
    if (this->labelSizeSlot.IsDirty()) {
        this->labelSize = this->labelSizeSlot.Param<core::param::FloatParam>()->Value();
        this->labelSizeSlot.ResetDirty();
    }

}
