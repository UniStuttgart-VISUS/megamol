/*
 * ImageViewer.cpp
 *
 * Copyright (C) 2010 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "vislib/graphics/gl/IncludeAllGL.h"
#include "imageviewer2/ImageViewer.h"
#include "mmcore/misc/PngBitmapCodec.h"
#include "imageviewer2/JpegBitmapCodec.h"
#include "vislib/graphics/BitmapCodecCollection.h"

//#define _USE_MATH_DEFINES
#include "mmcore/param/FilePathParam.h"
#include "mmcore/param/StringParam.h"
#include "mmcore/param/ButtonParam.h"
#include "mmcore/param/IntParam.h"
#include "mmcore/param/EnumParam.h"
#include "mmcore/view/CallRender3D.h"
#include "vislib/sys/Log.h"
#include "mmcore/CoreInstance.h"
#include "mmcore/view/AbstractView3D.h"
//#include <cmath>

using namespace megamol::core;
using namespace megamol;

/*
 * misc::ImageViewer::ImageViewer
 */
imageviewer2::ImageViewer::ImageViewer(void) : Renderer3DModule(),
        leftFilenameSlot("leftImg", "The image file name"),
        rightFilenameSlot("rightImg", "The image file name"),
        pasteFilenamesSlot("pasteFiles", "Slot to paste both file names at once (semicolon-separated)"),
        pasteSlideshowSlot("pasteShow", "Slot to paste filename pairs (semicolor-separated) on individual lines"),
        firstSlot("first", "go to first image in slideshow"),
        previousSlot("previous", "go to previous image in slideshow"),
        currentSlot("current", "current slideshow image index"),
        nextSlot("next", "go to next image in slideshow"),
        lastSlot("last", "go to last image in slideshow"),
        defaultEye("defaultEye", "where the image goes if the slideshow only has one image per line"),
        width(1), height(1), tiles(), leftFiles(), rightFiles() {

    this->leftFilenameSlot << new param::FilePathParam("");
    this->MakeSlotAvailable(&this->leftFilenameSlot);

    this->rightFilenameSlot << new param::FilePathParam("");
    this->MakeSlotAvailable(&this->rightFilenameSlot);

    this->pasteFilenamesSlot << new param::StringParam("");
    this->pasteFilenamesSlot.SetUpdateCallback(&ImageViewer::onFilesPasted);
    this->MakeSlotAvailable(&this->pasteFilenamesSlot);

    this->pasteSlideshowSlot << new param::StringParam("");
    this->pasteSlideshowSlot.SetUpdateCallback(&ImageViewer::onSlideshowPasted);
    this->MakeSlotAvailable(&this->pasteSlideshowSlot);

    this->firstSlot << new param::ButtonParam();
    this->firstSlot.SetUpdateCallback(&ImageViewer::onFirstPressed);
    this->MakeSlotAvailable(&this->firstSlot);
    this->previousSlot << new param::ButtonParam(264); // pageup
    this->previousSlot.SetUpdateCallback(&ImageViewer::onPreviousPressed);
    this->MakeSlotAvailable(&this->previousSlot);
    
    this->currentSlot << new param::IntParam(0);
    this->currentSlot.SetUpdateCallback(&ImageViewer::onCurrentSet);
    this->MakeSlotAvailable(&this->currentSlot);
    
    this->nextSlot << new param::ButtonParam(265); // pagedown
    this->nextSlot.SetUpdateCallback(&ImageViewer::onNextPressed);
    this->MakeSlotAvailable(&this->nextSlot);
    this->lastSlot << new param::ButtonParam();
    this->lastSlot.SetUpdateCallback(&ImageViewer::onLastPressed);
    this->MakeSlotAvailable(&this->lastSlot);

    param::EnumParam *ep = new param::EnumParam(0);
    ep->SetTypePair(0, "left");
    ep->SetTypePair(1, "right");
    this->defaultEye << ep;
    this->MakeSlotAvailable(&this->defaultEye);

    this->leftFiles.AssertCapacity(20);
    this->rightFiles.AssertCapacity(20);
    this->leftFiles.SetCapacityIncrement(20);
    this->rightFiles.SetCapacityIncrement(20);

}


/*
 * misc::ImageViewer::~ImageViewer
 */
imageviewer2::ImageViewer::~ImageViewer(void) {
    this->Release();
}


/*
 * misc::ImageViewer::create
 */
bool imageviewer2::ImageViewer::create(void) {
    // intentionally empty
    vislib::graphics::BitmapCodecCollection::DefaultCollection().AddCodec(new sg::graphics::PngBitmapCodec());
    vislib::graphics::BitmapCodecCollection::DefaultCollection().AddCodec(new sg::graphics::JpegBitmapCodec());

    return true;
}


/*
 * misc::ImageViewer::GetCapabilities
 */
bool imageviewer2::ImageViewer::GetCapabilities(Call& call) {
    view::CallRender3D *cr = dynamic_cast<view::CallRender3D*>(&call);
    if (cr == NULL) return false;

    cr->SetCapabilities(
        view::CallRender3D::CAP_RENDER
        );

    return true;
}


/*
 * imageviewer2::ImageViewer::GetExtents
 */
bool imageviewer2::ImageViewer::GetExtents(Call& call) {
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
 * imageviewer2::ImageViewer::release
 */
void imageviewer2::ImageViewer::release(void) {
//    this->image.Release();
}


void imageviewer2::ImageViewer::assertImage(bool rightEye) {
    param::ParamSlot *filenameSlot = rightEye ? (&this->rightFilenameSlot) : (&this->leftFilenameSlot);
    if (filenameSlot->IsDirty()) {
        filenameSlot->ResetDirty();
        const vislib::TString& filename = filenameSlot->Param<param::FilePathParam>()->Value();
        static vislib::graphics::BitmapImage img;
        //static ::sg::graphics::PngBitmapCodec codec;
        //codec.Image() = &img;
        static const unsigned int TILE_SIZE = 2 * 1024;
        ::glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
        try {
            //if (codec.Load(filename)) {
            if (vislib::graphics::BitmapCodecCollection::DefaultCollection().LoadBitmapImage(img, filename)) {
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
 * imageviewer2::ImageViewer::Render
 */
bool imageviewer2::ImageViewer::Render(Call& call) {
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
 * imageviewer2::ImageViewer::onFilesPasted
 */
bool imageviewer2::ImageViewer::onFilesPasted(param::ParamSlot &slot) {
    vislib::TString str(this->pasteFilenamesSlot.Param<param::StringParam>()->Value());
    vislib::TString left, right;
    str.Replace(_T("\r"), _T(""));
    this->interpretLine(str, left, right);
    this->leftFilenameSlot.Param<param::FilePathParam>()->SetValue(left);
    this->rightFilenameSlot.Param<param::FilePathParam>()->SetValue(right);
    return true;
}


void imageviewer2::ImageViewer::interpretLine(const vislib::TString source, vislib::TString& left, vislib::TString& right) {
    vislib::TString line(source);
    line.Replace(_T("\n"), _T(""));
    vislib::TString::Size scp = line.Find(_T(";"));
    if (scp != vislib::TString::INVALID_POS) {
        left = line.Substring(0, scp);
        right = line.Substring(scp + 1);
    } else {
        if (this->defaultEye.Param<param::EnumParam>()->Value() == 0) {
            left = line;
            right = _T("");
        } else {
            left = _T("");
            right = line;
        }
    }
    //line.Replace(_T("\n"), _T(""));
    //vislib::TString::Size scp = line.Find(_T(";"));
    //if (scp != vislib::TString::INVALID_POS) {
    //    this->leftFiles.Append(line.Substring(0, scp));
    //    this->rightFiles.Append(line.Substring(scp + 1));
    //} else {
    //    if (this->defaultEye.Param<param::EnumParam>()->Value() == 0) {
    //        this->leftFiles.Append(line);
    //        this->rightFiles.Append(_T(""));
    //    } else {
    //        this->rightFiles.Append(line);
    //        this->leftFiles.Append(_T(""));
    //    }
    //}
}

/*
* imageviewer2::ImageViewer::onSlideshowPasted
*/
bool imageviewer2::ImageViewer::onSlideshowPasted(param::ParamSlot &slot) {
    vislib::TString left, right;
    this->leftFiles.Clear();
    this->rightFiles.Clear();
    vislib::TString str(this->pasteSlideshowSlot.Param<param::StringParam>()->Value());
    str.Replace(_T("\r"), _T(""));
    vislib::TString::Size startPos = 0;
    vislib::TString::Size pos = str.Find(_T("\n"), startPos);
    while (pos != vislib::TString::INVALID_POS) {
        vislib::TString line = str.Substring(startPos, pos - startPos + 1);
        this->interpretLine(line, left, right);
        this->leftFiles.Append(left);
        this->rightFiles.Append(right);
        startPos = pos + 1;
        pos = str.Find(_T("\n"), startPos);
    }
    vislib::TString line = str.Substring(startPos);
    this->interpretLine(line, left, right);
    this->leftFiles.Append(left);
    this->rightFiles.Append(right);
    this->currentSlot.Param<param::IntParam>()->SetValue(-1);
    this->onFirstPressed(this->firstSlot);
    return true;
}

/*
* imageviewer2::ImageViewer::onFirstPressed
*/
bool imageviewer2::ImageViewer::onFirstPressed(param::ParamSlot &slot) {
    this->currentSlot.Param<param::IntParam>()->SetValue(0);
    return true;
}


/*
* imageviewer2::ImageViewer::onPreviousPressed
*/
bool imageviewer2::ImageViewer::onPreviousPressed(param::ParamSlot &slot) {
    this->currentSlot.Param<param::IntParam>()->SetValue(this->currentSlot.Param<param::IntParam>()->Value() - 1);
    return true;
}


/*
* imageviewer2::ImageViewer::onNextPressed
*/
bool imageviewer2::ImageViewer::onNextPressed(param::ParamSlot &slot) {
    this->currentSlot.Param<param::IntParam>()->SetValue(this->currentSlot.Param<param::IntParam>()->Value() + 1);
    return true;
}


/*
* imageviewer2::ImageViewer::onLastPressed
*/
bool imageviewer2::ImageViewer::onLastPressed(param::ParamSlot &slot) {
    this->currentSlot.Param<param::IntParam>()->SetValue(this->leftFiles.Count() - 1);
    return true;
}


/*
* imageviewer2::ImageViewer::onCurrentSet
*/
bool imageviewer2::ImageViewer::onCurrentSet(param::ParamSlot &slot) {
    int s = slot.Param<param::IntParam>()->Value();
    if (s > -1 && s < this->leftFiles.Count()) {
        this->leftFilenameSlot.Param<param::FilePathParam>()->SetValue(leftFiles[s]);
        this->rightFilenameSlot.Param<param::FilePathParam>()->SetValue(rightFiles[s]);

        // use ResetViewOnBBoxChange of your View! 

        //vislib::Stack<AbstractNamedObjectContainer::const_ptr_type> stack;
        //stack.Push(this->GetCoreInstance()->ModuleGraphRoot());
        //while (!stack.IsEmpty()) {
        //    AbstractNamedObjectContainer::const_ptr_type node = stack.Pop();
        //    AbstractNamedObjectContainer::child_list_type::const_iterator children, childrenend;
        //    childrenend = node->ChildList_End();
        //    for (children = node->ChildList_Begin(); children != childrenend; ++children) {
        //        AbstractNamedObject::const_ptr_type child = *children;
        //        AbstractNamedObjectContainer::const_ptr_type anoc = AbstractNamedObjectContainer::dynamic_pointer_cast(child);
        //        if (anoc) stack.Push(anoc); // continue
        //        const megamol::core::view::AbstractView3D::const_ptr_type vi = std::dynamic_pointer_cast<const megamol::core::view::AbstractView3D>(child);
        //        if (vi) {
        //            printf("found a view to guess");
        //        }
        //    }
        //}
    }
    return true;
}
