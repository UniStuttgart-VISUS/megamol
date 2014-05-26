/*
 * ImageViewer.cpp
 *
 * Copyright (C) 2010 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "glh/glh_genext.h"
//#define _USE_MATH_DEFINES
#include "ImageViewer.h"
#include "param/FilePathParam.h"
#include "param/StringParam.h"
#include "view/CallRender3D.h"
#include "vislib/Log.h"
//#include <cmath>

using namespace megamol::core;


/*
 * misc::ImageViewer::ImageViewer
 */
misc::ImageViewer::ImageViewer(void) : Renderer3DModule(),
        leftFilenameSlot("leftImg", "The image file name"),
        rightFilenameSlot("rightImg", "The image file name"),
        pasteFilenamesSlot("pasteFiles", "Slot to paste both file names at once (semicolon-separated)"),
        width(1), height(1), tiles() {

    this->leftFilenameSlot << new param::FilePathParam("");
    this->MakeSlotAvailable(&this->leftFilenameSlot);

    this->rightFilenameSlot << new param::FilePathParam("");
    this->MakeSlotAvailable(&this->rightFilenameSlot);

    this->pasteFilenamesSlot << new param::StringParam("");
    this->pasteFilenamesSlot.SetUpdateCallback(&ImageViewer::onFilesPasted);
    this->MakeSlotAvailable(&this->pasteFilenamesSlot);
}


/*
 * misc::ImageViewer::~ImageViewer
 */
misc::ImageViewer::~ImageViewer(void) {
    this->Release();
}


/*
 * misc::ImageViewer::create
 */
bool misc::ImageViewer::create(void) {
    // intentionally empty
    return true;
}


/*
 * misc::ImageViewer::GetCapabilities
 */
bool misc::ImageViewer::GetCapabilities(Call& call) {
    view::CallRender3D *cr = dynamic_cast<view::CallRender3D*>(&call);
    if (cr == NULL) return false;

    cr->SetCapabilities(
        view::CallRender3D::CAP_RENDER
        );

    return true;
}


/*
 * misc::ImageViewer::GetExtents
 */
bool misc::ImageViewer::GetExtents(Call& call) {
    view::CallRender3D *cr = dynamic_cast<view::CallRender3D*>(&call);
    if (cr == NULL) return false;

    if (cr->GetCameraParameters() != NULL) {
        bool rightEye = (cr->GetCameraParameters()->Eye() == vislib::graphics::CameraParameters::RIGHT_EYE);
        assertImage(rightEye);
    }

    cr->SetTimeFramesCount(1);
    cr->AccessBoundingBoxes().Clear();
    cr->AccessBoundingBoxes().SetObjectSpaceBBox(0.0f, 0.0f, -0.5f,
        static_cast<float>(this->width), static_cast<float>(this->height), 0.5f);
    cr->AccessBoundingBoxes().SetObjectSpaceClipBox(cr->AccessBoundingBoxes().ObjectSpaceBBox());
    cr->AccessBoundingBoxes().MakeScaledWorld(1.0f);

    return true;
}


/*
 * misc::ImageViewer::release
 */
void misc::ImageViewer::release(void) {
//    this->image.Release();
}


void misc::ImageViewer::assertImage(bool rightEye) {
    param::ParamSlot *filenameSlot = rightEye ? (&this->rightFilenameSlot) : (&this->leftFilenameSlot);
    if (filenameSlot->IsDirty()) {
        filenameSlot->ResetDirty();
        const vislib::TString& filename = filenameSlot->Param<param::FilePathParam>()->Value();
        static vislib::graphics::BitmapImage img;
        static ::sg::graphics::PngBitmapCodec codec;
        static const unsigned int TILE_SIZE = 2 * 1024;
        codec.Image() = &img;
        ::glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
        try {
            if (codec.Load(filename)) {
                img.Convert(vislib::graphics::BitmapImage::TemplateByteRGB);
                this->width = img.Width();
                this->height = img.Height();
                this->tiles.Clear();
                BYTE *buf = new BYTE[TILE_SIZE * TILE_SIZE * 3];
                for (unsigned int y = 0; y < this->height; y += TILE_SIZE) {
                    unsigned int h = vislib::math::Min(TILE_SIZE, this->height - y);
                    for (unsigned int x = 0; x < this->width; x += TILE_SIZE) {
                        unsigned int w = vislib::math::Min(TILE_SIZE, this->width - x);
                        for (unsigned int l = 0; l < h; l++) {
                            ::memcpy(buf + (l * w * 3), img.PeekDataAs<BYTE>() + ((y + l) * this->width * 3 + x * 3), w * 3);
                        }
                        this->tiles.Add(vislib::Pair<vislib::math::Rectangle<float>, vislib::SmartPtr<vislib::graphics::gl::OpenGLTexture2D> >());
                        this->tiles.Last().First().Set(static_cast<float>(x), static_cast<float>(this->height - y), static_cast<float>(x + w), static_cast<float>(this->height - (y + h)));
                        this->tiles.Last().SetSecond(new vislib::graphics::gl::OpenGLTexture2D());
                        if (this->tiles.Last().Second()->Create(w, h, false, buf, GL_RGB) != GL_NO_ERROR) {
                            this->tiles.RemoveLast();
                        } else {
                            this->tiles.Last().Second()->SetFilter(GL_LINEAR, GL_LINEAR);
                            this->tiles.Last().Second()->SetWrap(GL_CLAMP_TO_EDGE, GL_CLAMP_TO_EDGE);
                        }
                    }
                }
                delete[] buf;
                img.CreateImage(1, 1, vislib::graphics::BitmapImage::TemplateByteRGB);
            } else {
                printf("Failed: Load\n");
            }
        } catch(vislib::Exception ex) {
            printf("Failed: %s (%s;%d)\n", ex.GetMsgA(), ex.GetFile(), ex.GetLine());
        } catch(...) {
            printf("Failed\n");
        }
    }
}


/*
 * misc::ImageViewer::Render
 */
bool misc::ImageViewer::Render(Call& call) {
    view::CallRender3D *cr3d = dynamic_cast<view::CallRender3D*>(&call);
    if (cr3d == NULL) return false;
    bool rightEye = (cr3d->GetCameraParameters()->Eye() == vislib::graphics::CameraParameters::RIGHT_EYE);
    //param::ParamSlot *filenameSlot = rightEye ? (&this->rightFilenameSlot) : (&this->leftFilenameSlot);
    ::glEnable(GL_TEXTURE_2D);
    assertImage(rightEye);

    ::glDisable(GL_LINE_SMOOTH);
    ::glDisable(GL_BLEND);
    ::glDisable(GL_LIGHTING);
    ::glEnable(GL_DEPTH_TEST);
    ::glLineWidth(1.0f);
    ::glDisable(GL_LINE_SMOOTH);

    ::glColor3ub(255, 255, 255);
    for (SIZE_T i = 0; i < this->tiles.Count(); i++) {
        this->tiles[i].Second()->Bind();
        ::glBegin(GL_QUADS);
        ::glTexCoord2i(0, 0); ::glVertex2f(this->tiles[i].First().Left(), this->tiles[i].First().Bottom());
        ::glTexCoord2i(0, 1); ::glVertex2f(this->tiles[i].First().Left(), this->tiles[i].First().Top());
        ::glTexCoord2i(1, 1); ::glVertex2f(this->tiles[i].First().Right(), this->tiles[i].First().Top());
        ::glTexCoord2i(1, 0); ::glVertex2f(this->tiles[i].First().Right(), this->tiles[i].First().Bottom());
        ::glEnd();
    }
    ::glBindTexture(GL_TEXTURE_2D, 0);

    ::glDisable(GL_TEXTURE);

    return true;
}


/*
 * misc::ImageViewer::onFilesPasted
 */
bool misc::ImageViewer::onFilesPasted(param::ParamSlot &slot) {
    vislib::TString str(this->pasteFilenamesSlot.Param<param::StringParam>()->Value());
    vislib::TString::Size scp = str.Find(_T(";"));
    if (scp != vislib::TString::INVALID_POS) {
        this->leftFilenameSlot.Param<param::FilePathParam>()->SetValue(str.Substring(0, scp));
        this->rightFilenameSlot.Param<param::FilePathParam>()->SetValue(str.Substring(scp + 1));
    }
    return true;
}
