/*
* TestFontRenderer.cpp
*
* Copyright (C) 2010 by VISUS (Universitaet Stuttgart)
* Alle Rechte vorbehalten.
*/

#include "stdafx.h"

#include "mmcore/view/special/TestFontRenderer.h"

#include "mmcore/utility/ResourceWrapper.h"
#include "mmcore/misc/PngBitmapCodec.h"
#include "mmcore/view/CallRender3D.h"
#include "mmcore/CoreInstance.h"
#include "mmcore/view/AbstractView3D.h"
#include "mmcore/view/special/Verdana.inc"

#include "mmcore/param/FilePathParam.h"
#include "mmcore/param/FloatParam.h"
#include "mmcore/param/ButtonParam.h"

#include "vislib/graphics/gl/Verdana.inc"
#include "vislib/graphics/gl/IncludeAllGL.h"
#include "vislib/sys/Log.h"
#include "vislib/sys/File.h"

using namespace megamol;
using namespace megamol::core;
using namespace megamol::core::view;
using namespace megamol::core::view::special;


/*
* TestFontRenderer::TestFontRenderer
*/
TestFontRenderer::TestFontRenderer(void) : Renderer3DModule(),
    simpleFont(),
    outlineFont(vislib::graphics::gl::FontInfo_Verdana, vislib::graphics::gl::OutlineFont::RENDERTYPE_FILL),
    sdfFont(SDFFont::BitmapFont::BMFONT_EVOLVENTA, SDFFont::RENDERTYPE_FILL),
    paramRenderMode1("01_RenderMode-01", "... ."),
    paramRenderMode2("02_RenderMode-02", "... ."),
    paramRenderMode3("03_RenderMode-03", "... ."),
    paramRenderMode4("04_RenderMode-04", "... .")
    {


    this->paramRenderMode1.SetParameter(new param::ButtonParam('1'));
    this->MakeSlotAvailable(&this->paramRenderMode1);

    this->paramRenderMode2.SetParameter(new param::ButtonParam('2'));
    this->MakeSlotAvailable(&this->paramRenderMode2);

    this->paramRenderMode3.SetParameter(new param::ButtonParam('3'));
    this->MakeSlotAvailable(&this->paramRenderMode3);

    this->paramRenderMode4.SetParameter(new param::ButtonParam('4'));
    this->MakeSlotAvailable(&this->paramRenderMode4);


    this->renderMode = 1;
}


/*
* TestFontRenderer::TestFontRenderer
*/
TestFontRenderer::~TestFontRenderer(void) {
    this->Release();
}


/*
* TestFontRenderer::release
*/
void TestFontRenderer::release(void) {


}


/*
* TestFontRenderer::create
*/
bool TestFontRenderer::create(void) {

    // Initialise simple font
    if (!this->simpleFont.Initialise()) {
        vislib::sys::Log::DefaultLog.WriteError("[TestFontRenderer] [create] Couldn't initialize the simple font.");
        return false;
    }
    // Initialise outline font
    if (!this->outlineFont.Initialise()) {
        vislib::sys::Log::DefaultLog.WriteError("[TestFontRenderer] [create] Couldn't initialize the outline font.");
        return false;
    }
    // Initialise sdf font
    if (!this->sdfFont.Initialise()) {
        vislib::sys::Log::DefaultLog.WriteError("[TestFontRenderer] [create] Couldn't initialize the sdf font.");
        return false;
    }

    return true;
}

/*
* TestFontRenderer::GetCapabilities
*/
bool TestFontRenderer::GetCapabilities(Call& call) {
    CallRender3D *cr3d = dynamic_cast<CallRender3D*>(&call);
    if (cr3d == NULL) return false;


    return true;
}

/*
* TestFontRenderer::GetExtents
*/
bool TestFontRenderer::GetExtents(core::Call& call) {

    megamol::core::view::CallRender3D *cr2d = dynamic_cast<view::CallRender3D*>(&call);
    if (cr2d == NULL) {
        vislib::sys::Log::DefaultLog.WriteError("[TestFontRenderer] [GetExtents] Call is NULL.");
        return false;
    }

    // Unused

    return true;
}


/*
* TestFontRenderer::Render
*/
bool TestFontRenderer::Render(core::Call& call) {

    megamol::core::view::CallRender3D *cr2d = dynamic_cast<megamol::core::view::CallRender3D*>(&call);
    if (cr2d == NULL) {
        vislib::sys::Log::DefaultLog.WriteError("[TestFontRenderer] [Render] Call is NULL.");
        return false;
    }

    if (this->paramRenderMode1.IsDirty()) {
        this->renderMode = 1;
        this->paramRenderMode1.ResetDirty();
    }
    if (this->paramRenderMode2.IsDirty()) {
        this->renderMode = 2;
        this->paramRenderMode2.ResetDirty();
    }
    if (this->paramRenderMode3.IsDirty()) {
        this->renderMode = 3;
        this->paramRenderMode3.ResetDirty();
    }
    if (this->paramRenderMode4.IsDirty()) {
        this->renderMode = 4;
        this->paramRenderMode4.ResetDirty();
    }

    // ------------------------------------------------------------------------
    if (this->renderMode == 1) {

        // Get current color for shader
        glColor4f(0.0f, 0.5f, 0.0f, 1.0f);

        this->sdfFont.draw();

    }
    // ------------------------------------------------------------------------
    else if (this->renderMode == 2) {

        // Opengl setup 

        // Get the foreground color (inverse background color)
        float bgColor[4];
        float fgColor[4] = { 1.0f, 1.0f, 1.0f, 1.0f };
        glGetFloatv(GL_COLOR_CLEAR_VALUE, bgColor);
        for (unsigned int i = 0; i < 3; i++) {
            fgColor[i] -= bgColor[i];
        }

        // Get current viewport
        int vp[4];
        glGetIntegerv(GL_VIEWPORT, vp);
        int   vpWidth = vp[2] - vp[0];
        int   vpHeight = vp[3] - vp[1];
        float vpH      = static_cast<float>(vpHeight);
        float vpW      = static_cast<float>(vpWidth);

        GLfloat tmpLw;
        glGetFloatv(GL_LINE_WIDTH, &tmpLw);

        glHint(GL_LINE_SMOOTH_HINT, GL_NICEST);
        glHint(GL_POLYGON_SMOOTH_HINT, GL_NICEST);

        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

        glDisableClientState(GL_VERTEX_ARRAY);
        glDisableClientState(GL_COLOR_ARRAY);

        glDisable(GL_CULL_FACE);
        glDisable(GL_LIGHTING);
        glDisable(GL_DEPTH_TEST);

        glDisable(GL_TEXTURE_2D);
        glDisable(GL_TEXTURE_1D);

        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        glEnable(GL_BLEND);

        glMatrixMode(GL_PROJECTION);
        glPushMatrix();
        glLoadIdentity();
        glOrtho(0.0f, vpW, 0.0f, vpH, -1.0, 1.0);

        glMatrixMode(GL_MODELVIEW);
        glPushMatrix();
        glLoadIdentity();
 

        // ------------------------------------------------------------------------

        float fontSize = vpH*0.5f; // 50% of viewport height
        float nol      = 6.0f; // number of lines

        vislib::StringA simpleString  = "The Simple Font. ";
        vislib::StringA filledString  = "The Filled Font. ";
        vislib::StringA outlineString = "The Outline Font. ";
        vislib::StringA sdfString     = "The SDF Font. ";

        glColor4fv(fgColor);

        // Adapt font size
        vislib::StringA tmpString = "--------------------------";
        // Adapt to width
        float maxWidthFontSize = fontSize;
        while (this->simpleFont.LineWidth(maxWidthFontSize, tmpString) > vpW) {
            //maxWidthFontSize -= 0.1f;
            maxWidthFontSize -= 1.0f;
        }
        // Adapt to height 
        float maxHeightFontSize = fontSize;
        while (nol*this->simpleFont.LineHeight(maxHeightFontSize) > vpH) {
            //maxHeightFontSize -= 0.1f;
            maxHeightFontSize -= 1.0f;
        }
        fontSize = (maxWidthFontSize < maxHeightFontSize) ? (maxWidthFontSize) : (maxHeightFontSize);
        fontSize = (fontSize < 0.0f) ? (0.1f) : (fontSize);


        // SIMPLE FONT 
        float simpleWidth = this->simpleFont.LineWidth(fontSize, simpleString);
        this->simpleFont.DrawString(0.0f, vpH, simpleWidth, 1.0f, fontSize, true, simpleString, vislib::graphics::AbstractFont::ALIGN_LEFT_TOP);


        // OUTLINE FONT 
        this->outlineFont.SetRenderType(vislib::graphics::gl::OutlineFont::RENDERTYPE_OUTLINE);
        float outlineWidth = this->outlineFont.LineWidth(fontSize, outlineString);
        this->outlineFont.DrawString(0.0f, vpH - (fontSize*1.0f), outlineWidth, 1.0f, fontSize, true, outlineString, vislib::graphics::AbstractFont::ALIGN_LEFT_TOP);

        glEnable(GL_LINE_SMOOTH);
        outlineString += "AA ";
        outlineWidth = this->outlineFont.LineWidth(fontSize, outlineString);
        this->outlineFont.DrawString(0.0f, vpH - (fontSize*2.0f), outlineWidth, 1.0f, fontSize, true, outlineString, vislib::graphics::AbstractFont::ALIGN_LEFT_TOP);
        glDisable(GL_LINE_SMOOTH);


        // FILLED FONT 
        this->outlineFont.SetRenderType(vislib::graphics::gl::OutlineFont::RENDERTYPE_FILL);
        outlineWidth = this->outlineFont.LineWidth(fontSize, outlineString);
        this->outlineFont.DrawString(0.0f, vpH - (fontSize*3.0f), outlineWidth, 1.0f, fontSize, true, outlineString, vislib::graphics::AbstractFont::ALIGN_LEFT_TOP);

        glEnable(GL_POLYGON_SMOOTH);
        filledString += "AA ";
        outlineWidth = this->outlineFont.LineWidth(fontSize, filledString);
        this->outlineFont.DrawString(0.0f, vpH - (fontSize*4.0f), outlineWidth, 1.0f, fontSize, true, filledString, vislib::graphics::AbstractFont::ALIGN_LEFT_TOP);
        glDisable(GL_POLYGON_SMOOTH);

        // SDF FONT
        //float sdfWidth = this->sdfFont.LineWidth(fontSize, sdfString);
        //this->sdfFont.DrawString(0.0f, vpH - (fontSize*5.0f), sdfWidth, 1.0f, fontSize, true, sdfString, megamol::core::view::special::AbstractFont::ALIGN_LEFT_TOP);

        glPopMatrix();
        glMatrixMode(GL_PROJECTION);
        glPopMatrix();
        glMatrixMode(GL_MODELVIEW);

        // Reset opengl 
        glLineWidth(tmpLw);
        glDisable(GL_BLEND);
        glDisable(GL_LINE_SMOOTH);
        glDisable(GL_POLYGON_SMOOTH);
    }
    // ------------------------------------------------------------------------
    else if (this->renderMode == 3) {



    }
    // ------------------------------------------------------------------------
    else if (this->renderMode == 4) {



    }
    // ------------------------------------------------------------------------
    else {
        vislib::sys::Log::DefaultLog.WriteWarn("[TestFontRenderer] [render] Unknown render mode ...");
    }


    return true;
}

