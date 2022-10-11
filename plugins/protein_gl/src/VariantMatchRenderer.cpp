//
// VariantMatchRenderer.cpp
//
// Copyright (C) 2013 by University of Stuttgart (VISUS).
// All rights reserved.
//
// Created on: Jun 15, 2013
//     Author: scharnkn
//

#include "VariantMatchRenderer.h"
#include "ogl_error_check.h"
#include "protein_calls/VariantMatchDataCall.h"

#include "mmcore/CoreInstance.h"
#include "mmcore/param/FloatParam.h"
#include "mmcore/param/IntParam.h"
#include "mmcore/utility/log/Log.h"
#include "mmcore_gl/utility/ShaderFactory.h"
#include "vislib_gl/graphics/gl/OutlineFont.h"
#include "vislib_gl/graphics/gl/SimpleFont.h"
#include <algorithm>

using namespace megamol;
using namespace megamol::protein_gl;
using megamol::core::utility::log::Log;

#pragma push_macro("min")
#undef min
#pragma push_macro("max")
#undef max

/*
 * VariantMatchRenderer::VariantMatchRenderer
 */
VariantMatchRenderer::VariantMatchRenderer(void)
        : mmstd_gl::Renderer2DModuleGL()
        , dataCallerSlot("getData", "Connects the rendering with data storage")
        , minColSlot("minCol", "...")
        , maxColSlot("maxCol", "...")
        , matrixTex(0)
        , thefont(vislib_gl::graphics::gl::FontInfo_Verdana) {

    // Data caller slot to get matching matrix
    this->dataCallerSlot.SetCompatibleCall<protein_calls::VariantMatchDataCallDescription>();
    this->MakeSlotAvailable(&this->dataCallerSlot);

    // Parameter for minimum color
    this->minCol = 0.0f;
    this->minColSlot.SetParameter(new core::param::FloatParam(this->minCol));
    this->MakeSlotAvailable(&this->minColSlot);

    // Parameter for maximum color
    this->maxCol = 1.0f;
    this->maxColSlot.SetParameter(new core::param::FloatParam(this->maxCol));
    this->MakeSlotAvailable(&this->maxColSlot);
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
    auto const shader_options = msf::ShaderFactoryOptionsOpenGL(GetCoreInstance()->GetShaderPaths());

    try {
        matrixTexShader = core::utility::make_glowl_shader("matrixTexShader", shader_options,
            "protein_gl/2dplot/variantMatrix.vert.glsl", "protein_gl/2dplot/variantMatrix.frag.glsl");

        colorMapShader = core::utility::make_glowl_shader("colorMapShader", shader_options,
            "protein_gl/2dplot/variantMatrix_CM.vert.glsl", "protein_gl/2dplot/variantMatrix_CM.frag.glsl");

    } catch (std::exception& e) {
        Log::DefaultLog.WriteError(("VariantMatchRenderer: " + std::string(e.what())).c_str());
        return false;
    }

    return true;
}


/*
 * VariantMatchRenderer::GetExtents
 */
bool VariantMatchRenderer::GetExtents(mmstd_gl::CallRender2DGL& call) {
    call.AccessBoundingBoxes().SetBoundingBox(-1.0f, -1.0f, 0, 1.0f, 1.0f, 0);
    return true;
}


/*
 * VariantMatchRenderer::release
 */
void VariantMatchRenderer::release(void) {
    if (this->matrixTex) {
        glDeleteTextures(1, &this->matrixTex);
    }
    CheckForGLError();
}


/*
 * VariantMatchRenderer::Render
 */
bool VariantMatchRenderer::Render(mmstd_gl::CallRender2DGL& call) {

    // float gridHalfStep;

    if (!this->thefont.Initialise()) {
        return false;
    }

    // Update parameters
    this->updateParams();

    // Get pointer to VariantMatchDataCall
    protein_calls::VariantMatchDataCall* vmc = this->dataCallerSlot.CallAs<protein_calls::VariantMatchDataCall>();
    if (vmc == NULL) {
        return false;
    }

    //  Execute call for Getdata
    if (!(*vmc)(protein_calls::VariantMatchDataCall::CallForGetData)) {
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
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, vmc->GetVariantCnt(), vmc->GetVariantCnt(), 0, GL_ALPHA, GL_FLOAT,
        vmc->GetMatch());

    // // DEBUG Print matrix values
    // printf("Variant count %u\n", vmc->GetVariantCnt());
    // for (size_t i = 0; i < vmc->GetVariantCnt(); ++i) {
    //     for (size_t j = 0; j < vmc->GetVariantCnt(); ++j) {
    //         printf("%.2f ", vmc->GetMatch()[j * vmc->GetVariantCnt() + i]);
    //     }
    //     printf("\n");
    // }
    // // END DEBUG

    //    ::glMatrixMode(GL_PROJECTION);
    //    ::glPushMatrix();
    //    ::glLoadIdentity();

    //    ::glMatrixMode(GL_MODELVIEW);
    //    ::glPushMatrix();
    //    ::glLoadIdentity();

    // gridStep = (2.0 - this->labelSpace)/static_cast<float>(vmc->GetVariantCnt());
    // gridHalfStep = gridStep*0.5f;

    //    printf("Min %f, max %f\n", vmc->GetMin(), vmc->GetMax());

    // Draw matrix texture
    //    ::glColor3f(0.0f, 0.6f, 0.6f);
    this->matrixTexShader->use();
    glUniform1iARB(this->matrixTexShader->getUniformLocation("matrixTex"), 0);
    glUniform1fARB(this->matrixTexShader->getUniformLocation("minVal"), this->minCol);
    glUniform1fARB(this->matrixTexShader->getUniformLocation("maxVal"), this->maxCol);
    ::glBegin(GL_QUADS);
    ::glTexCoord2f(1.0f, 0.0f);
    ::glVertex2f(-1.0, -1.0);
    ::glTexCoord2f(1.0f, 1.0f);
    ::glVertex2f(1.0, -1.0);
    ::glTexCoord2f(0.0f, 1.0f);
    ::glVertex2f(1.0f, 1.0f);
    ::glTexCoord2f(0.0f, 0.0f);
    ::glVertex2f(-1.0, 1.0f);
    ::glEnd();
    glUseProgram(0);

    // Draw color map
    if (!this->drawColorMap()) {
        return false;
    }

    glDisable(GL_TEXTURE_2D);

    ::glColor3f(1.0f, 1.0f, 1.0f);
    ::glLineWidth(3);
    // Draw vertical lines
    float gridStep = 2.0f / static_cast<float>(vmc->GetVariantCnt());
    ::glBegin(GL_LINES);
    for (int i = 0; i < static_cast<int>(vmc->GetVariantCnt()); ++i) {
        ::glVertex2f(gridStep * i - 1.0f, -1.0f);
        ::glVertex2f(gridStep * i - 1.0f, 1.0f);
    }
    ::glEnd();

    // Draw horizontal lines
    ::glBegin(GL_LINES);
    for (int i = 1; i <= static_cast<int>(vmc->GetVariantCnt()); ++i) {
        ::glVertex2f(-1.0, gridStep * i - 1.0f);
        ::glVertex2f(1.0, gridStep * i - 1.0f);
    }
    ::glEnd();


    glColor3f(0.0, 0.0, 0.0);
    //    vislib_gl::graphics::gl::SimpleFont f;
    vislib::StringA str;
    this->fontSize = std::min(2.0f / static_cast<float>(vmc->GetVariantCnt()), 0.1f);
    float lineHeight = this->thefont.LineHeight(2.0f / static_cast<float>(vmc->GetVariantCnt()));
    float maxLineWidth = 0.0f;
    for (int i = 0; i < static_cast<int>(vmc->GetVariantCnt()); ++i) {
        maxLineWidth = std::max(maxLineWidth, this->thefont.LineWidth(fontSize, vmc->GetLabels()[i].PeekBuffer()));
    }
    maxLineWidth += this->thefont.LineWidth(fontSize, "000 : ");

    //    f.SetSize(0.5f);
    // Draw vertical labels
    for (int i = 0; i < static_cast<int>(vmc->GetVariantCnt()); ++i) {
        // printf("LABEL %s\n", vmc->GetLabels()[i].PeekBuffer());
        str.Format("%3i : ", i);
        str.Append(vmc->GetLabels()[i].PeekBuffer());
        this->thefont.DrawString(-1.0f - maxLineWidth, // Left coordinate of the rectangle
            1.0f - lineHeight * (i + 1),               // Upper coordinate of the rectangle
            maxLineWidth,                              // The width of the rectangle
            lineHeight,                                // The height of the rectangle
            fontSize,                                  // The font size
            true,                                      // Flip y
            // vmc->GetLabels()[i].PeekBuffer(),    // The label
            str.PeekBuffer(), vislib::graphics::AbstractFont::ALIGN_LEFT_MIDDLE);
    }
    // Draw horizontal labels
    ::glMatrixMode(GL_MODELVIEW);
    for (int i = 0; i < static_cast<int>(vmc->GetVariantCnt()); ++i) {
        ::glPushMatrix();
        //        ::glLoadIdentity();
        //        ::glTranslatef(
        //                (this->labelSpace+gridStep*(i))-1.0 + gridHalfStep,
        //                1.0 - this->labelSpace*0.5,
        //                0.0);
        ::glRotatef(90, 0.0, 0.0, 1.0);
        str.Format("%3i : ", i);
        str.Append(vmc->GetLabels()[i].PeekBuffer());
        this->thefont.DrawString(1.0f,   // Left coordinate of the rectangle
            1.0f - lineHeight * (i + 1), // Upper coordinate of the rectangle
            maxLineWidth,                // The width of the rectangle
            lineHeight,                  // The height of the rectangle
            fontSize,                    // The font size
            true,                        // Flip y
            // vmc->GetLabels()[i].PeekBuffer(),    // The label
            str.PeekBuffer(), // The label
            vislib::graphics::AbstractFont::ALIGN_LEFT_MIDDLE);
        ::glPopMatrix();
    }

    //    ::glPopMatrix();
    //
    //    ::glMatrixMode(GL_PROJECTION);
    //    ::glPopMatrix();


    return true;
}


/*
 * VariantMatchRenderer::drawColorMap
 */
bool VariantMatchRenderer::drawColorMap() {

    // Draw color gradient

    this->colorMapShader->use();
    glUniform1fARB(this->colorMapShader->getUniformLocation("minVal"), this->minCol);
    glUniform1fARB(this->colorMapShader->getUniformLocation("maxVal"), this->maxCol);

    ::glBegin(GL_QUADS);
    ::glVertex2f(-1.0f, -1.2f);
    ::glVertex2f(1.0f, -1.2f);
    ::glVertex2f(1.0f, -1.1f);
    ::glVertex2f(-1.0f, -1.1f);
    ::glEnd();
    glUseProgram(0);

    ::glBegin(GL_LINE_STRIP);
    ::glVertex2f(-1.0f, -1.2f);
    ::glVertex2f(1.0f, -1.2f);
    ::glVertex2f(1.0f, -1.1f);
    ::glVertex2f(-1.0f, -1.1f);
    ::glVertex2f(-1.0f, -1.2f);
    ::glEnd();

    // Draw labels

    vislib::StringA str;
    str.Format("%.3f", this->minCol);
    this->thefont.DrawString(-1.0f, // Left coordinate of the rectangle
        -1.32f,                     // Upper coordinate of the rectangle
        this->fontSize,             // The font size
        true,                       // Flip y
        str.PeekBuffer(), vislib::graphics::AbstractFont::ALIGN_CENTER_BOTTOM);

    str.Format("%.3f", this->maxCol);
    this->thefont.DrawString(1.0f, // Left coordinate of the rectangle
        -1.32f,                    // Upper coordinate of the rectangle
        this->fontSize,            // The font size
        true,                      // Flip y
        str.PeekBuffer(), vislib::graphics::AbstractFont::ALIGN_CENTER_BOTTOM);

    return true;
}


/*
 * VariantMatchRenderer::updateParams
 */
void VariantMatchRenderer::updateParams() {

    // Min color
    if (this->minColSlot.IsDirty()) {
        this->minCol = this->minColSlot.Param<core::param::FloatParam>()->Value();
        this->minColSlot.ResetDirty();
    }

    // Max color
    if (this->maxColSlot.IsDirty()) {
        this->maxCol = this->maxColSlot.Param<core::param::FloatParam>()->Value();
        this->maxColSlot.ResetDirty();
    }
}
