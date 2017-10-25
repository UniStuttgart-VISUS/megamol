/*
* WatermarkRenderer.cpp
*
* Copyright (C) 2010 by VISUS (Universitaet Stuttgart)
* Alle Rechte vorbehalten.
*/

#include "stdafx.h"

#include "mmcore/misc/WatermarkRenderer.h"

#include "mmcore/utility/ResourceWrapper.h"
#include "mmcore/misc/PngBitmapCodec.h"
#include "mmcore/view/CallRender3D.h"
#include "mmcore/CoreInstance.h"
#include "mmcore/view/AbstractView3D.h"

#include "mmcore/param/FilePathParam.h"
#include "mmcore/param/FloatParam.h"

#include "vislib/graphics/gl/IncludeAllGL.h"
#include "vislib/sys/Log.h"

using namespace megamol;
using namespace megamol::core;
using namespace megamol::core::misc;

/*
* WatermarkRenderer::WatermarkRenderer
*/
WatermarkRenderer::WatermarkRenderer(void) : Renderer3DModule(),
    paramImgTopLeft(      "01_imageTopLeft", "The image file name for the top left watermark."),
    paramImgTopRight(     "02_imageTopLeft", "The image file name for the top right watermark."),
    paramImgBottomLeft(   "03_imageTopLeft", "The image file name for the bottom left watermark."),
    paramImgBottomRight(  "04_imageTopLeft", "The image file name for the bottom right watermark."),
    paramScaleAll(        "05_scaleAll", "The scale factor for all images."),
    paramScaleTopLeft(    "06_scaleTopLeft", "The scale factor for the top left watermark."),
    paramScaleTopRight(   "07_scaleTopRight", "The scale factor for the top rightwatermark."),
    paramScaleBottomLeft( "08_scaleBottomLeft", "The scale factor for the botttom left watermark."),
    paramScaleBottomRight("09_scaleBottomRight", "The scale factor for the bottom right watermark."),
    paramAlpha(           "10_alpha", "The alpha value for the watermarks.")
    {

    /* Init image file name params */
    this->paramImgTopLeft.SetParameter(new param::FilePathParam(""));
    this->MakeSlotAvailable(&this->paramImgTopLeft);

    this->paramImgTopRight.SetParameter(new param::FilePathParam(""));
    this->MakeSlotAvailable(&this->paramImgTopRight);

    this->paramImgBottomLeft.SetParameter(new param::FilePathParam(""));
    this->MakeSlotAvailable(&this->paramImgBottomLeft);

    this->paramImgBottomRight.SetParameter(new param::FilePathParam(""));
    this->MakeSlotAvailable(&this->paramImgBottomRight);

    /* Init scale params */
    this->paramScaleAll.SetParameter(new param::FloatParam(1.0, 0.0f));
    this->MakeSlotAvailable(&this->paramScaleAll);

    this->paramScaleTopLeft.SetParameter(new param::FloatParam(1.0, 0.0f));
    this->MakeSlotAvailable(&this->paramScaleTopLeft);

    this->paramScaleTopRight.SetParameter(new param::FloatParam(1.0, 0.0f));
    this->MakeSlotAvailable(&this->paramScaleTopRight);

    this->paramScaleBottomLeft.SetParameter(new param::FloatParam(1.0, 0.0f));
    this->MakeSlotAvailable(&this->paramScaleBottomLeft);

    this->paramScaleBottomRight.SetParameter(new param::FloatParam(1.0, 0.0f));
    this->MakeSlotAvailable(&this->paramScaleBottomRight);

    /* Init alpha param */
    this->paramAlpha.SetParameter(new param::FloatParam(1.0f, 0.0f, 1.0f));
    this->MakeSlotAvailable(&this->paramAlpha);
}


/*
* WatermarkRenderer::WatermarkRenderer
*/
WatermarkRenderer::~WatermarkRenderer(void) {
    this->Release();
}


/*
* WatermarkRenderer::create
*/
bool WatermarkRenderer::create(void) {
    // intentionally empty

    return true;
}


/*
* WatermarkRenderer::GetCapabilities
*/
bool WatermarkRenderer::GetCapabilities(Call& call) {

    view::CallRender3D *cr = dynamic_cast<view::CallRender3D*>(&call);
    if (cr == NULL) {
        vislib::sys::Log::DefaultLog.WriteError("[WatermarkRenderer] [GetCapabilities] Call is NULL.");
        return false;
    }

    cr->SetCapabilities(view::CallRender3D::CAP_RENDER);

    return true;
}


/*
* WatermarkRenderer::GetExtents
*/
bool WatermarkRenderer::GetExtents(Call& call) {

    view::CallRender3D *cr = dynamic_cast<view::CallRender3D*>(&call);
    if (cr == NULL) {
        vislib::sys::Log::DefaultLog.WriteError("[WatermarkRenderer] [GetExtents] Call is NULL.");
        return false;
    }

    // Unused

    return true;
}


/*
* WatermarkRenderer::release
*/
void WatermarkRenderer::release(void) {
    // intentionally empty
}


/*
* WatermarkRenderer::Render
*/
bool WatermarkRenderer::Render(Call& call) {

    view::CallRender3D *cr3d = dynamic_cast<view::CallRender3D*>(&call);
    if (cr3d == NULL) {
        vislib::sys::Log::DefaultLog.WriteError("[WatermarkRenderer] [Render] Call is NULL.");
        return false;
    }

    /* Update parameters */
    if (this->paramImgTopLeft.IsDirty()) {
        this->paramImgTopLeft.ResetDirty();
        this->loadTexture(WatermarkRenderer::TOP_LEFT, static_cast<vislib::StringA>(this->paramImgTopLeft.Param<param::FilePathParam>()->Value()));
    }
    if (this->paramImgTopRight.IsDirty()) {
        this->paramImgTopRight.ResetDirty();
        this->loadTexture(WatermarkRenderer::TOP_RIGHT, static_cast<vislib::StringA>(this->paramImgTopRight.Param<param::FilePathParam>()->Value()));
    }
    if (this->paramImgBottomLeft.IsDirty()) {
        this->paramImgBottomLeft.ResetDirty();
        this->loadTexture(WatermarkRenderer::BOTTOM_LEFT, static_cast<vislib::StringA>(this->paramImgBottomLeft.Param<param::FilePathParam>()->Value()));
    }
    if (this->paramImgBottomRight.IsDirty()) {
        this->paramImgBottomRight.ResetDirty();
        this->loadTexture(WatermarkRenderer::BOTTOM_RIGHT, static_cast<vislib::StringA>(this->paramImgBottomRight.Param<param::FilePathParam>()->Value()));
    }
    if (this->paramScaleAll.IsDirty()) {
        this->paramScaleAll.ResetDirty();
        float scaleAll = this->paramScaleAll.Param<param::FloatParam>()->Value();
        this->paramScaleTopLeft.Param<param::FloatParam>()->SetValue(scaleAll, false);
        this->paramScaleTopRight.Param<param::FloatParam>()->SetValue(scaleAll, false);
        this->paramScaleBottomLeft.Param<param::FloatParam>()->SetValue(scaleAll, false);
        this->paramScaleBottomRight.Param<param::FloatParam>()->SetValue(scaleAll, false);
    }

    // Get current viewport
    int vp[4];
    glGetIntegerv(GL_VIEWPORT, vp);
    float vpWidth  = static_cast<float>(vp[2] - vp[0]);
    float vpHeight = static_cast<float>(vp[3] - vp[1]);

    // OpenGl states
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    glDisable(GL_LIGHTING);
    glDisable(GL_CULL_FACE);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glEnable(GL_BLEND);

    // Set matrices
    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();
    glOrtho(0.0f, vpWidth, 0.0f, vpHeight, -1.0, 1.0);
    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glLoadIdentity();

    // Draw help text in front of all
    glTranslatef(0.0f, 0.0f, 1.0f);

    // Render watermarks ...
    this->renderWatermark(WatermarkRenderer::TOP_LEFT, vpHeight, vpWidth);
    this->renderWatermark(WatermarkRenderer::TOP_RIGHT, vpHeight, vpWidth);
    this->renderWatermark(WatermarkRenderer::BOTTOM_LEFT, vpHeight, vpWidth);
    this->renderWatermark(WatermarkRenderer::BOTTOM_RIGHT, vpHeight, vpWidth);

    // Reset matrices
    glPopMatrix();
    glMatrixMode(GL_PROJECTION);
    glPopMatrix();
    glMatrixMode(GL_MODELVIEW);

    // Reset OpenGl states
    glDisable(GL_BLEND);

    return true;
}


/*
* WatermarkRenderer::renderWatermark
*/
bool WatermarkRenderer::renderWatermark(WatermarkRenderer::corner cor, float vpH, float vpW) {

    // Temp
    float imageWidth = 200.0f;
    float imageHeight = 100.0f;

    /*

    // the scaling factors
    this->paramScaleTopLeft.Param<param::FloatParam>()->Value();
    this->paramScaleTopRight.Param<param::FloatParam>()->Value();
    this->paramScaleBottomLeft.Param<param::FloatParam>()->Value();
    this->paramScaleBottomRight.Param<param::FloatParam>()->Value();

    // the alpha value
    this->paramAlpha.Param<param::FloatParam>()->Value();

    */


    // Set watermark dimensions
    float left, top, bottom, right;
    vislib::graphics::gl::OpenGLTexture2D *tex = NULL;

    switch (cor) {
    case(WatermarkRenderer::TOP_LEFT):
        left = 0.0f;
        right = imageWidth;
        top = vpH;
        bottom = vpH - imageHeight;
        tex = &this->textureTopLeft;
        break;
    case(WatermarkRenderer::TOP_RIGHT):
        left = vpW - imageWidth;
        right = vpW;
        top = vpH;
        bottom = vpH - imageHeight;
        tex = &this->textureTopRight;
        break;
    case(WatermarkRenderer::BOTTOM_LEFT):
        left = 0.0f;
        right = imageWidth;
        top = imageHeight;
        bottom = 0.0f;
        tex = &this->textureBottomLeft;
        break;
    case(WatermarkRenderer::BOTTOM_RIGHT):
        left = vpW - imageWidth;
        right = vpW;
        top = imageHeight;
        bottom = 0.0f;
        tex = &this->textureBottomRight;
        break;
    default: vislib::sys::Log::DefaultLog.WriteError("[WatermarkRenderer] [renderWatermark] Unknown corner - BUG.");
        break;
    }

    // Draw watermark texture
    if (tex->IsValid()) {
        glEnable(GL_TEXTURE_2D);
        tex->Bind();
    }
    else {
        glDisable(GL_TEXTURE_2D);
        vislib::sys::Log::DefaultLog.WriteError("[WatermarkRenderer] [renderWatermark] Texture is not valid - BUG.");
        return false;
    }
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    glBegin(GL_QUADS);
    glTexCoord2f(0.0f, 0.0f); glVertex2f(left, top);
    glTexCoord2f(1.0f, 0.0f); glVertex2f(right, top);
    glTexCoord2f(1.0f, 1.0f); glVertex2f(right, bottom);
    glTexCoord2f(0.0f, 1.0f); glVertex2f(left, bottom);
    glEnd();

    glDisable(GL_TEXTURE_2D);

    return true;
}


/*
* WatermarkRenderer::loadTexture
*/
bool WatermarkRenderer::loadTexture(WatermarkRenderer::corner cor, vislib::StringA filename) {

    if (!filename.IsEmpty()) {

        vislib::graphics::gl::OpenGLTexture2D *tex = NULL;

        switch (cor) {
        case(WatermarkRenderer::TOP_LEFT):
            tex = &this->textureTopLeft;
            break;
        case(WatermarkRenderer::TOP_RIGHT):
            tex = &this->textureTopRight;
            break;
        case(WatermarkRenderer::BOTTOM_LEFT):
            tex = &this->textureBottomLeft;
            break;
        case(WatermarkRenderer::BOTTOM_RIGHT):
            tex = &this->textureBottomRight;
            break;
        default: vislib::sys::Log::DefaultLog.WriteError("[WatermarkRenderer] [loadTexture] Unknown corner - BUG.");
            break;
        }

        if (tex == NULL) {
            vislib::sys::Log::DefaultLog.WriteError("[WatermarkRenderer] [loadTexture] Texture pointer is NULL - BUG.");
            return false;
        }

        static vislib::graphics::BitmapImage img;
        static sg::graphics::PngBitmapCodec  pbc;
        pbc.Image() = &img;
        ::glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
        void *buf = NULL;
        SIZE_T size = 0;

        if ((size = megamol::core::utility::ResourceWrapper::LoadResource(
            this->GetCoreInstance()->Configuration(), filename, &buf)) > 0) {
            if (pbc.Load(buf, size)) {
                img.Convert(vislib::graphics::BitmapImage::TemplateByteRGBA);
                for (unsigned int i = 0; i < img.Width() * img.Height(); i++) {
                    BYTE r = img.PeekDataAs<BYTE>()[i * 4 + 0];
                    BYTE g = img.PeekDataAs<BYTE>()[i * 4 + 1];
                    BYTE b = img.PeekDataAs<BYTE>()[i * 4 + 2];
                    if (r + g + b > 0) {
                        img.PeekDataAs<BYTE>()[i * 4 + 3] = 255;
                    }
                    else {
                        img.PeekDataAs<BYTE>()[i * 4 + 3] = 0;
                    }
                }
                tex =  new vislib::graphics::gl::OpenGLTexture2D();
                if (tex->Create(img.Width(), img.Height(), false, img.PeekDataAs<BYTE>(), GL_RGBA) != GL_NO_ERROR) {
                    vislib::sys::Log::DefaultLog.WriteError("[WatermarkRenderer] [loadTexture] Could not load \"%s\" texture.", filename.PeekBuffer());
                    ARY_SAFE_DELETE(buf);
                    return false;
                }
                tex->SetFilter(GL_LINEAR, GL_LINEAR);
                ARY_SAFE_DELETE(buf);
                return true;
            }
            else {
                vislib::sys::Log::DefaultLog.WriteError("[WatermarkRenderer] [loadTexture] Could not read \"%s\" texture.", filename.PeekBuffer());
            }
        }
        else {
            vislib::sys::Log::DefaultLog.WriteError("[WatermarkRenderer] [loadTexture] Could not find \"%s\" texture.", filename.PeekBuffer());
        }
        return false;
    }
    else {
        vislib::sys::Log::DefaultLog.WriteWarn("[WatermarkRenderer] [loadTexture] Filename is empty.");
    }

    return true;
}
