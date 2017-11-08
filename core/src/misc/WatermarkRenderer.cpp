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
#include "vislib/sys/File.h"

using namespace megamol;
using namespace megamol::core;
using namespace megamol::core::misc;


/*
* WatermarkRenderer::WatermarkRenderer
*/
WatermarkRenderer::WatermarkRenderer(void) : Renderer3DModule(),
    paramAlpha(           "01_alpha", "The alpha value for the watermarks."),
    paramScaleAll(        "02_scaleAll", "The scale factor for all images."),
    paramImgTopLeft(      "03_imageTopLeft", "The image file name for the top left watermark."),
    paramScaleTopLeft(    "04_scaleTopLeft", "The scale factor for the top left watermark."),
    paramImgTopRight(     "05_imageTopRight", "The image file name for the top right watermark."),
    paramScaleTopRight(   "06_scaleTopRight", "The scale factor for the top right watermark."),
    paramImgBottomLeft(   "07_imageBottomLeft", "The image file name for the bottom left watermark."),
    paramScaleBottomLeft( "08_scaleBottomLeft", "The scale factor for the botttom left watermark."),
    paramImgBottomRight(  "09_imageBottomRight", "The image file name for the bottom right watermark."),
    paramScaleBottomRight("10_scaleBottomRight", "The scale factor for the bottom right watermark."),
    textureBottomLeft(), textureBottomRight(), textureTopLeft(), textureTopRight()
    {

    // Init image file name params
    this->paramImgTopLeft.SetParameter(new param::FilePathParam(""));
    this->MakeSlotAvailable(&this->paramImgTopLeft);

    this->paramImgTopRight.SetParameter(new param::FilePathParam(""));
    this->MakeSlotAvailable(&this->paramImgTopRight);

    this->paramImgBottomLeft.SetParameter(new param::FilePathParam(""));
    this->MakeSlotAvailable(&this->paramImgBottomLeft);

    this->paramImgBottomRight.SetParameter(new param::FilePathParam(""));
    this->MakeSlotAvailable(&this->paramImgBottomRight);

    // Init scale params
    this->paramScaleAll.SetParameter(new param::FloatParam(1.0));
    this->MakeSlotAvailable(&this->paramScaleAll);

    this->paramScaleTopLeft.SetParameter(new param::FloatParam(1.0, 0.0000001f));
    this->MakeSlotAvailable(&this->paramScaleTopLeft);

    this->paramScaleTopRight.SetParameter(new param::FloatParam(1.0, 0.0000001f));
    this->MakeSlotAvailable(&this->paramScaleTopRight);

    this->paramScaleBottomLeft.SetParameter(new param::FloatParam(1.0, 0.0000001f));
    this->MakeSlotAvailable(&this->paramScaleBottomLeft);

    this->paramScaleBottomRight.SetParameter(new param::FloatParam(1.0, 0.0000001f));
    this->MakeSlotAvailable(&this->paramScaleBottomRight);

    // Init alpha param 
    this->paramAlpha.SetParameter(new param::FloatParam(1.0f, 0.0f, 1.0f));
    this->MakeSlotAvailable(&this->paramAlpha);

    // Init variables 
    this->lastScaleAll     = 1.0f;
    this->firstParamChange = false;
}


/*
* WatermarkRenderer::WatermarkRenderer
*/
WatermarkRenderer::~WatermarkRenderer(void) {
    this->Release();
}


/*
* WatermarkRenderer::release
*/
void WatermarkRenderer::release(void) {
    //this->textureBottomLeft.Release();
    //this->textureBottomRight.Release();
    //this->textureTopLeft.Release();
    //this->textureTopRight.Release();
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
* WatermarkRenderer::Render
*/
bool WatermarkRenderer::Render(Call& call) {

    view::CallRender3D *cr3d = dynamic_cast<view::CallRender3D*>(&call);
    if (cr3d == NULL) {
        vislib::sys::Log::DefaultLog.WriteError("[WatermarkRenderer] [Render] Call is NULL.");
        return false;
    }

    // Update parameters 
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

        // Ignore first usage of scaleAll to set lastScaleAll when parameter value is loaded
        if (this->firstParamChange) {
            this->paramScaleTopLeft.Param<param::FloatParam>()->SetValue(this->paramScaleTopLeft.Param<param::FloatParam>()->Value() + (scaleAll - this->lastScaleAll), false);
            this->paramScaleTopRight.Param<param::FloatParam>()->SetValue(this->paramScaleTopRight.Param<param::FloatParam>()->Value() + (scaleAll - this->lastScaleAll), false);
            this->paramScaleBottomLeft.Param<param::FloatParam>()->SetValue(this->paramScaleBottomLeft.Param<param::FloatParam>()->Value() + (scaleAll - this->lastScaleAll), false);
            this->paramScaleBottomRight.Param<param::FloatParam>()->SetValue(this->paramScaleBottomRight.Param<param::FloatParam>()->Value() + (scaleAll - this->lastScaleAll), false);
        }
        else {
            this->firstParamChange = true;
        }
        this->lastScaleAll = scaleAll;
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

    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LEQUAL);

    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    // Set matrices
    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();
    glOrtho(0.0f, vpWidth, 0.0f, vpHeight, -1.0, 1.0);

    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glLoadIdentity();
    // Draw textures in front of all
    glTranslatef(0.0f, 0.0f, 1.0f);

    // Render watermarks 
    glEnable(GL_TEXTURE_2D);
    this->renderWatermark(WatermarkRenderer::TOP_LEFT, vpHeight, vpWidth);
    this->renderWatermark(WatermarkRenderer::TOP_RIGHT, vpHeight, vpWidth);
    this->renderWatermark(WatermarkRenderer::BOTTOM_LEFT, vpHeight, vpWidth);
    this->renderWatermark(WatermarkRenderer::BOTTOM_RIGHT, vpHeight, vpWidth);
    glDisable(GL_TEXTURE_2D);

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

    // Set watermark dimensions
    float imageWidth, imageHeight;
    float left, top, bottom, right;
    float scale;
    vislib::graphics::gl::OpenGLTexture2D *tex = NULL;
    float alpha = this->paramAlpha.Param<param::FloatParam>()->Value();

    // Set default image width ... e.g. depending on viewport size
    float fixImgWidth = imageWidth; // vpW* 1.0f;

    switch (cor) {
    case(WatermarkRenderer::TOP_LEFT):
        tex          = &this->textureTopLeft;
        scale        = this->paramScaleTopLeft.Param<param::FloatParam>()->Value();
        imageWidth   = fixImgWidth * scale;
        imageHeight  = fixImgWidth * (this->sizeTopLeft.Y() / this->sizeTopLeft.X()) * scale;
        left         = 0.0f;
        right        = imageWidth;
        top          = vpH;
        bottom       = vpH - imageHeight;
        break;
    case(WatermarkRenderer::TOP_RIGHT):
        tex          = &this->textureTopRight;
        scale        = this->paramScaleTopRight.Param<param::FloatParam>()->Value();
        imageWidth   = fixImgWidth * scale;
        imageHeight  = fixImgWidth * (this->sizeTopRight.Y() / this->sizeTopRight.X()) * scale;
        left         = vpW - imageWidth;
        right        = vpW;
        top          = vpH;
        bottom       = vpH - imageHeight;
        break;
    case(WatermarkRenderer::BOTTOM_LEFT):
        tex          = &this->textureBottomLeft;
        scale        = this->paramScaleBottomLeft.Param<param::FloatParam>()->Value();
        imageWidth   = fixImgWidth * scale;
        imageHeight  = fixImgWidth * (this->sizeBottomLeft.Y() / this->sizeBottomLeft.X()) * scale;
        left         = 0.0f;
        right        = imageWidth;
        top          = imageHeight;
        bottom       = 0.0f;
        break;
    case(WatermarkRenderer::BOTTOM_RIGHT):
        tex          = &this->textureBottomRight;
        scale        = this->paramScaleBottomRight.Param<param::FloatParam>()->Value();
        imageWidth   = fixImgWidth * scale;
        imageHeight  = fixImgWidth * (this->sizeBottomRight.Y() / this->sizeBottomRight.X()) * scale;
        left         = vpW - imageWidth;
        right        = vpW;
        top          = imageHeight;
        bottom       = 0.0f;
        break;
    default: vislib::sys::Log::DefaultLog.WriteError("[WatermarkRenderer] [renderWatermark] Unknown corner - BUG.");
        break;
    }

    // Draw watermark texture
    if (tex->IsValid()) {
        tex->Bind();
        glColor4f(1.0f, 1.0f, 1.0f, alpha);
        glBegin(GL_QUADS);
            glTexCoord2f(0.0f, 0.0f); glVertex2f(left, top);
            glTexCoord2f(1.0f, 0.0f); glVertex2f(right, top);
            glTexCoord2f(1.0f, 1.0f); glVertex2f(right, bottom);
            glTexCoord2f(0.0f, 1.0f); glVertex2f(left, bottom);
        glEnd();
    }
    else {
        return false;
    }

    return true;
}


/*
* WatermarkRenderer::loadTexture
*/
bool WatermarkRenderer::loadTexture(WatermarkRenderer::corner cor, vislib::StringA filename) {

    if (!filename.IsEmpty()) {

        vislib::graphics::gl::OpenGLTexture2D *tex     = NULL;
        vislib::math::Vector<float, 2>        *texSize = NULL;

        switch (cor) {
        case(WatermarkRenderer::TOP_LEFT):
            tex     = &this->textureTopLeft;
            texSize = &this->sizeTopLeft;
            break;
        case(WatermarkRenderer::TOP_RIGHT):
            tex     = &this->textureTopRight;
            texSize = &this->sizeTopRight;
            break;
        case(WatermarkRenderer::BOTTOM_LEFT):
            tex     = &this->textureBottomLeft;
            texSize = &this->sizeBottomLeft;
            break;
        case(WatermarkRenderer::BOTTOM_RIGHT):
            tex     = &this->textureBottomRight;
            texSize = &this->sizeBottomRight;
            break;
        default: vislib::sys::Log::DefaultLog.WriteError("[WatermarkRenderer] [renderWatermark] Unknown corner - BUG.");
            break;
        }

        if (tex->IsValid()) {
            tex->Release();
        }

        static vislib::graphics::BitmapImage img;
        static sg::graphics::PngBitmapCodec  pbc;
        pbc.Image() = &img;
        ::glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
        void *buf = NULL;
        SIZE_T size = 0;

        // If textures should be loaded as resources:  if ((size = megamol::core::utility::ResourceWrapper::LoadResource(this->GetCoreInstance()->Configuration(), filename, &buf)) > 0) {
        if ((size = this->loadFile(filename, &buf)) > 0) {
            if (pbc.Load(buf, size)) {
                img.Convert(vislib::graphics::BitmapImage::TemplateByteRGBA);
                texSize->SetX(img.Width());
                texSize->SetY(img.Height());
                // Set alpha to zero for black background
                //for (unsigned int i = 0; i < img.Width() * img.Height(); i++) {
                //    BYTE r = img.PeekDataAs<BYTE>()[i * 4 + 0];
                //    BYTE g = img.PeekDataAs<BYTE>()[i * 4 + 1];
                //    BYTE b = img.PeekDataAs<BYTE>()[i * 4 + 2];
                //    if (r + g + b > 0) {
                //        img.PeekDataAs<BYTE>()[i * 4 + 3] = 255;
                //    }
                //    else {
                //        img.PeekDataAs<BYTE>()[i * 4 + 3] = 0;
                //    }
                //}
                if (tex->Create(img.Width(), img.Height(), false, img.PeekDataAs<BYTE>(), GL_RGBA) != GL_NO_ERROR) {
                    vislib::sys::Log::DefaultLog.WriteError("[WatermarkRenderer] [loadTexture] Could not load \"%s\" texture.", filename.PeekBuffer());
                    ARY_SAFE_DELETE(buf);
                    return false;
                }
                tex->Bind();
                glGenerateMipmap(GL_TEXTURE_2D);
                glBindTexture(GL_TEXTURE_2D, 0);
                tex->SetFilter(GL_LINEAR_MIPMAP_LINEAR, GL_LINEAR);
                tex->SetWrap(GL_CLAMP_TO_EDGE, GL_CLAMP_TO_EDGE);
                ARY_SAFE_DELETE(buf);
                return true;
            }
            else {
                vislib::sys::Log::DefaultLog.WriteError("[WatermarkRenderer] [loadTexture] Could not read \"%s\" texture.", filename.PeekBuffer());
            }
        }
        else {
            // Error is already catched by loadFile function
            //vislib::sys::Log::DefaultLog.WriteError("[WatermarkRenderer] [loadTexture] Could not find \"%s\" texture.", filename.PeekBuffer());
        }
        return false;
    }
    else {
        vislib::sys::Log::DefaultLog.WriteWarn("[WatermarkRenderer] [loadTexture] Unable to load file: No filename given\n");
    }

    return true;
}


/*
* WatermarkRenderer::loadTexture
*
* Based on: megamol::core::utility::ResourceWrapper::LoadResource() but without the lookup in the resource folder(s)
*/
SIZE_T WatermarkRenderer::loadFile(vislib::StringA name, void **outData) {

    *outData = NULL;

    vislib::StringW filename = static_cast<vislib::StringW>(name);
    if (filename.IsEmpty()) {
        vislib::sys::Log::DefaultLog.WriteError("[WatermarkRenderer] [loadFile] Unable to load file: No filename given\n");
        return 0;
    }

    if (!vislib::sys::File::Exists(filename)) {
        vislib::sys::Log::DefaultLog.WriteError("[WatermarkRenderer] [loadFile] Unable to load file \"%s\": Not existing\n", name.PeekBuffer());
        return 0;
    }

    SIZE_T size = static_cast<SIZE_T>(vislib::sys::File::GetSize(filename));
    if (size < 1) {
        vislib::sys::Log::DefaultLog.WriteError("[WatermarkRenderer] [loadFile] Unable to load file \"%s\": File is empty\n", name.PeekBuffer());
        return 0;
    }

    vislib::sys::FastFile f;
    if (!f.Open(filename, vislib::sys::File::READ_ONLY, vislib::sys::File::SHARE_READ, vislib::sys::File::OPEN_ONLY)) {
        vislib::sys::Log::DefaultLog.WriteError("[WatermarkRenderer] [loadFile] Unable to load file \"%s\": Cannot open file\n", name.PeekBuffer());
        return 0;
    }

    *outData = new BYTE[size];
    SIZE_T num = static_cast<SIZE_T>(f.Read(*outData, size));
    if (num != size) {
        vislib::sys::Log::DefaultLog.WriteError("[WatermarkRenderer] [loadFile] Unable to load file \"%s\": Cannot read whole file\n", name.PeekBuffer());
        ARY_SAFE_DELETE(*outData);
        return 0;
    }

    return num;
}
