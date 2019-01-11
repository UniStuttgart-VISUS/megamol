/*
 * ImageViewer.cpp
 *
 * Copyright (C) 2010 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "imageviewer2/ImageViewer.h"
#include "imageviewer2/JpegBitmapCodec.h"
#include "mmcore/misc/PngBitmapCodec.h"
#include "vislib/graphics/BitmapCodecCollection.h"
#include "vislib/graphics/gl/IncludeAllGL.h"

//#define _USE_MATH_DEFINES
#include "image_calls/Image2DCall.h"
#include "mmcore/CoreInstance.h"
#include "mmcore/cluster/mpi/MpiCall.h"
#include "mmcore/param/ButtonParam.h"
#include "mmcore/param/EnumParam.h"
#include "mmcore/param/FilePathParam.h"
#include "mmcore/param/IntParam.h"
#include "mmcore/param/StringParam.h"
#include "mmcore/view/AbstractRenderingView.h"
#include "mmcore/view/CallRender3D.h"
#include "vislib/sys/Log.h"
#include "vislib/sys/SystemInformation.h"
//#include <cmath>

using namespace megamol::core;
using namespace megamol;

/*
 * misc::ImageViewer::ImageViewer
 */
imageviewer2::ImageViewer::ImageViewer(void)
    : Renderer3DModule()
    , leftFilenameSlot("leftImg", "The image file name")
    , rightFilenameSlot("rightImg", "The image file name")
    , pasteFilenamesSlot("pasteFiles", "Slot to paste both file names at once (semicolon-separated)")
    , pasteSlideshowSlot("pasteShow", "Slot to paste filename pairs (semicolor-separated) on individual lines")
    , firstSlot("first", "go to first image in slideshow")
    , previousSlot("previous", "go to previous image in slideshow")
    , currentSlot("current", "current slideshow image index")
    , nextSlot("next", "go to next image in slideshow")
    , lastSlot("last", "go to last image in slideshow")
    , blankMachine("blankMachine", "semicolon-separated list of machines that do not load image")
    , defaultEye("defaultEye", "where the image goes if the slideshow only has one image per line")
    , callRequestMpi("requestMpi", "Requests initialisation of MPI and the communicator for the view.")
    , callRequestImage{"requestImage", "Requests an image to display"}
    , width(1)
    , height(1)
    , tiles()
    , leftFiles()
    , rightFiles()
    , datahash{std::numeric_limits<size_t>::max()} {

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

    param::EnumParam* ep = new param::EnumParam(0);
    ep->SetTypePair(0, "left");
    ep->SetTypePair(1, "right");
    this->defaultEye << ep;
    this->MakeSlotAvailable(&this->defaultEye);

    this->leftFiles.AssertCapacity(20);
    this->rightFiles.AssertCapacity(20);
    this->leftFiles.SetCapacityIncrement(20);
    this->rightFiles.SetCapacityIncrement(20);

    this->blankMachine << new param::StringParam("");
    this->blankMachine.SetUpdateCallback(&ImageViewer::onBlankMachineSet);
    this->MakeSlotAvailable(&this->blankMachine);

    vislib::sys::SystemInformation::ComputerName(this->machineName);
    this->machineName.ToLowerCase();

    this->callRequestMpi.SetCompatibleCall<cluster::mpi::MpiCallDescription>();
    this->MakeSlotAvailable(&this->callRequestMpi);

    this->callRequestImage.SetCompatibleCall<image_calls::Image2DCallDescription>();
    this->MakeSlotAvailable(&this->callRequestImage);

}


/*
 * misc::ImageViewer::~ImageViewer
 */
imageviewer2::ImageViewer::~ImageViewer(void) { this->Release(); }


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
 * imageviewer2::ImageViewer::GetExtents
 */
bool imageviewer2::ImageViewer::GetExtents(Call& call) {
    view::CallRender3D* cr = dynamic_cast<view::CallRender3D*>(&call);
    if (cr == NULL) return false;

    if (cr->GetCameraParameters() != NULL) {
        bool rightEye = (cr->GetCameraParameters()->Eye() == vislib::graphics::CameraParameters::RIGHT_EYE);
        if (!assertImage(rightEye)) return false;
    }

    cr->SetTimeFramesCount(1);
    cr->AccessBoundingBoxes().Clear();
    cr->AccessBoundingBoxes().SetObjectSpaceBBox(
        0.0f, 0.0f, -0.5f, static_cast<float>(this->width), static_cast<float>(this->height), 0.5f);
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


/*
 * imageviewer2::ImageViewer::assertImage
 */
bool imageviewer2::ImageViewer::assertImage(bool rightEye) {
    static bool registered = false;

    bool beBlank = this->blankMachines.Contains(this->machineName);
    bool useMpi = initMPI();
#ifdef WITH_MPI
    // generate communicators for each role
    if (useMpi && !registered) {
        myRole = beBlank ? IMG_BLANK : (rightEye ? IMG_RIGHT : IMG_LEFT);
        MPI_Comm_rank(this->comm, &rank);
        MPI_Comm_split(this->comm, myRole, 0, &roleComm);
        MPI_Comm_rank(roleComm, &roleRank);
        MPI_Comm_size(roleComm, &roleSize);
        vislib::sys::Log::DefaultLog.WriteInfo("ImageViewer2: role %s (rank %i of %i)",
            myRole == IMG_BLANK ? "blank" : (myRole == IMG_RIGHT ? "right" : "left"), roleRank, roleSize);
        registered = true;
    }
#endif /* WITH_MPI */


    bool imgcConnected = false;
    auto imgc = this->callRequestImage.CallAs<image_calls::Image2DCall>();
    if (imgc != nullptr) imgcConnected = true;
    uint8_t imgc_enc = megamol::image_calls::Image2DCall::Encoding::RAW;
    if (imgcConnected) {
        imgc_enc = imgc->GetEncoding();
    }

    param::ParamSlot* filenameSlot = rightEye ? (&this->rightFilenameSlot) : (&this->leftFilenameSlot);
    if (filenameSlot->IsDirty() || (imgcConnected /* && imgc->DataHash() != datahash*/) || useMpi) { //< imgc has precedence
        if (!imgcConnected) {
            filenameSlot->ResetDirty();
        }
        const vislib::TString& filename = filenameSlot->Param<param::FilePathParam>()->Value();
        static vislib::graphics::BitmapImage img;
        // static ::sg::graphics::PngBitmapCodec codec;
        // codec.Image() = &img;
        static const unsigned int TILE_SIZE = 2 * 1024;
        ::glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
        try {
            static bool handIsShaken = false;
#ifdef WITH_MPI
            if (useMpi && !handIsShaken) {
                handIsShaken = true;
                vislib::sys::Log::DefaultLog.WriteInfo("ImageViewer: IMGC Handshake\n");
                // handshake who has imgc
                int* imgcRes = nullptr;
                if (roleRank == 0) {
                    imgcRes = new int[roleSize];
                }

                int imgcCon = imgcConnected ? 1 : 0;
                MPI_Gather(&imgcCon, 1, MPI_INT, imgcRes, 1, MPI_INT, 0, roleComm);

                if (roleRank == 0) {
                    for (int i = 0; i < roleSize; ++i) {
                        if (imgcRes[i] == 1) {
                            roleImgcRank = i;
                            break;
                        }
                    }
                    delete[] imgcRes;
                }

                MPI_Bcast(&roleImgcRank, 1, MPI_INT, 0, roleComm);
                if (roleImgcRank != -1) {
                    remoteness = true;
                } else {
                    roleImgcRank = 0;
                }
                vislib::sys::Log::DefaultLog.WriteInfo(
                    "ImageViewer: IMGC Handshake result remoteness = %d imgcRank = %d\n", remoteness, roleImgcRank);
            }
#endif /* WITH_MPI */

            if (!beBlank && ((loadedFile != filename) || remoteness)) {
                int fileSize = 0;
                BYTE* allFile = nullptr;
                BYTE* imgc_data_ptr = nullptr;
#ifdef WITH_MPI
                // single node or role boss loads the image
                if (!useMpi || roleRank == roleImgcRank) {
                    vislib::sys::Log::DefaultLog.WriteInfo("ImageViewer2: role %s (rank %i of %i) loads an image",
                        myRole == IMG_BLANK ? "blank" : (myRole == IMG_RIGHT ? "right" : "left"), roleRank, roleSize);
#endif /* WITH_MPI */
                    if (!remoteness) {
                        vislib::sys::Log::DefaultLog.WriteInfo("ImageViewer: Loading file '%s' from disk\n", filename.PeekBuffer());
                        vislib::sys::FastFile in;
                        if (in.Open(filename, vislib::sys::File::READ_ONLY, vislib::sys::File::SHARE_READ,
                            vislib::sys::File::OPEN_ONLY)) {
                            fileSize = in.GetSize();
                            allFile = new BYTE[fileSize];
                            in.Read(allFile, fileSize);
                            in.Close();
                        } else {
                            printf("ImageViewer2: failed opening file\n");
                            fileSize = 0;
                        }
                    } else if (roleRank == roleImgcRank) {
                        vislib::sys::Log::DefaultLog.WriteInfo("ImageViewer: Retrieving image from call\n");
                        // retrieve data from call
                        if (!(*imgc)(0)) return false;
                        this->width = imgc->GetWidth();
                        this->height = imgc->GetHeight();
                        fileSize = imgc->GetFilesize();
                        allFile = reinterpret_cast<BYTE*>(imgc->GetData());
                    }
#ifdef WITH_MPI
                }
                // cluster nodes broadcast file size
                if (useMpi) {
                    int bcastRoot = roleImgcRank;
                    vislib::sys::Log::DefaultLog.WriteInfo("ImageViewer: Broadcast root = %d\n", bcastRoot);
                    MPI_Bcast(&fileSize, 1, MPI_INT, bcastRoot, roleComm);
                    vislib::sys::Log::DefaultLog.WriteInfo("ImageViewer2: rank %i of %i (role %s) knows fileSize = %i",
                        roleRank, roleSize, myRole == IMG_BLANK ? "blank" : (myRole == IMG_LEFT ? "left" : "right"),
                        fileSize);
                    if (roleRank != 0) {
                        allFile = new BYTE[fileSize];
                        vislib::sys::Log::DefaultLog.WriteInfo(
                            "ImageViewer2: rank %i of %i (role %s) late allocated file storage", roleRank, roleSize,
                            myRole == IMG_BLANK ? "blank" : (myRole == IMG_LEFT ? "left" : "right"));
                    }
                    // everyone gets the compressed file now
                    MPI_Bcast(allFile, fileSize, MPI_BYTE, bcastRoot, roleComm);
                    if (remoteness) {
                        MPI_Bcast(&this->width, 1, MPI_UNSIGNED, roleImgcRank, roleComm);
                        MPI_Bcast(&this->height, 1, MPI_UNSIGNED, roleImgcRank, roleComm);
                        MPI_Bcast(&imgc_enc, 1, MPI_UNSIGNED_CHAR, roleImgcRank, roleComm);
                    }
                }
#endif /* WITH_MPI */
                BYTE* image_ptr = nullptr;
                if (!remoteness) {
                    vislib::sys::Log::DefaultLog.WriteInfo("ImageViewer: Decoding Image\n");
                    if (vislib::graphics::BitmapCodecCollection::DefaultCollection().LoadBitmapImage(
                            img, allFile, fileSize)) {
                        img.Convert(vislib::graphics::BitmapImage::TemplateByteRGB);
                        this->width = img.Width();
                        this->height = img.Height();
                        image_ptr = img.PeekDataAs<BYTE>();
                    } else {
                        printf("ImageViewer2 failed decoding file\n");
                        fileSize = 0;
                        this->width = this->height = 0;
                    }
                } else {
                    vislib::sys::Log::DefaultLog.WriteInfo("ImageViewer: Decoding IMGC at rank %d\n", roleRank);
                    switch (imgc_enc) {
                        case image_calls::Image2DCall::BMP:
                        {
                            if (vislib::graphics::BitmapCodecCollection::DefaultCollection().LoadBitmapImage(
                                img, allFile, fileSize)) {
                                img.Convert(vislib::graphics::BitmapImage::TemplateByteRGB);
                                this->width = img.Width();
                                this->height = img.Height();
                                image_ptr = img.PeekDataAs<BYTE>();
                            } else {
                                printf("ImageViewer2 failed decoding file\n");
                                fileSize = 0;
                                this->width = this->height = 0;
                            }
                        } break;
                        case image_calls::Image2DCall::SNAPPY:
                        {
                        } break;
                        case image_calls::Image2DCall::RAW:
                        default:
                        {
                            image_ptr = allFile;
                        }
                    }
                }

                // now everyone should have a copy of the loaded image

                this->tiles.Clear();
                if (this->width > 0 && this->height > 0 && image_ptr != nullptr) {
                    BYTE* buf = new BYTE[TILE_SIZE * TILE_SIZE * 3];
                    for (unsigned int y = 0; y < this->height; y += TILE_SIZE) {
                        unsigned int h = vislib::math::Min(TILE_SIZE, this->height - y);
                        for (unsigned int x = 0; x < this->width; x += TILE_SIZE) {
                            unsigned int w = vislib::math::Min(TILE_SIZE, this->width - x);
                            for (unsigned int l = 0; l < h; l++) {
                                ::memcpy(buf + (l * w * 3), image_ptr + ((y + l) * this->width * 3 + x * 3), w * 3);
                            }
                            this->tiles.Add(vislib::Pair<vislib::math::Rectangle<float>,
                                vislib::SmartPtr<vislib::graphics::gl::OpenGLTexture2D>>());
                            this->tiles.Last().First().Set(static_cast<float>(x), static_cast<float>(this->height - y),
                                static_cast<float>(x + w), static_cast<float>(this->height - (y + h)));
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
                    if (!imgcConnected) {
                        delete[] allFile;
                    }
                    // img.CreateImage(1, 1, vislib::graphics::BitmapImage::TemplateByteRGB);
                    loadedFile = filename;
                }
#ifdef WITH_MPI
                // we finish this together
                if (useMpi) {
                    vislib::sys::Log::DefaultLog.WriteInfo(
                        "ImageViewer2: rank %i of %i (role %s) entering sync barrier", roleRank, roleSize,
                        myRole == IMG_BLANK ? "blank" : (myRole == IMG_LEFT ? "left" : "right"));
                    MPI_Barrier(roleComm);
                    vislib::sys::Log::DefaultLog.WriteInfo(
                        "ImageViewer2: rank %i of %i (role %s) leaving sync barrier", roleRank, roleSize,
                        myRole == IMG_BLANK ? "blank" : (myRole == IMG_LEFT ? "left" : "right"));
                }
#endif
            }

        } catch (vislib::Exception ex) {
            printf("Failed: %s (%s;%d)\n", ex.GetMsgA(), ex.GetFile(), ex.GetLine());
            return false;
        } catch (...) {
            printf("Failed\n");
            return false;
        }
    }
    return true;
}


bool imageviewer2::ImageViewer::initMPI() {
    bool retval = false;
#ifdef WITH_MPI
    if (this->comm == MPI_COMM_NULL) {
        auto c = this->callRequestMpi.CallAs<cluster::mpi::MpiCall>();
        if (c != nullptr) {
            /* New method: let MpiProvider do all the stuff. */
            if ((*c)(cluster::mpi::MpiCall::IDX_PROVIDE_MPI)) {
                vislib::sys::Log::DefaultLog.WriteInfo("Got MPI communicator.");
                this->comm = c->GetComm();
            } else {
                vislib::sys::Log::DefaultLog.WriteError(_T("Could not ")
                                                        _T("retrieve MPI communicator for the MPI-based view ")
                                                        _T("from the registered provider module."));
            }
        }

        if (this->comm != MPI_COMM_NULL) {
            vislib::sys::Log::DefaultLog.WriteInfo(_T("MPI is ready, ")
                                                   _T("retrieving communicator properties ..."));
            ::MPI_Comm_rank(this->comm, &this->mpiRank);
            ::MPI_Comm_size(this->comm, &this->mpiSize);
            vislib::sys::Log::DefaultLog.WriteInfo(_T("This view on %hs is %d ")
                                                   _T("of %d."),
                vislib::sys::SystemInformation::ComputerNameA().PeekBuffer(), this->mpiRank, this->mpiSize);
        } /* end if (this->comm != MPI_COMM_NULL) */
    }     /* end if (this->comm == MPI_COMM_NULL) */

    /* Determine success of the whole operation. */
    retval = (this->comm != MPI_COMM_NULL);
#endif /* WITH_MPI */
    return retval;
}


/*
 * imageviewer2::ImageViewer::Render
 */
bool imageviewer2::ImageViewer::Render(Call& call) {
    view::CallRender3D* cr3d = dynamic_cast<view::CallRender3D*>(&call);
    if (cr3d == NULL) return false;
    bool rightEye = (cr3d->GetCameraParameters()->Eye() == vislib::graphics::CameraParameters::RIGHT_EYE);
    // param::ParamSlot *filenameSlot = rightEye ? (&this->rightFilenameSlot) : (&this->leftFilenameSlot);
    ::glEnable(GL_TEXTURE_2D);
    if (!assertImage(rightEye)) return false;

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
        ::glTexCoord2i(0, remoteness ? 1 : 0);
        ::glVertex2f(this->tiles[i].First().Left(), this->tiles[i].First().Bottom());
        ::glTexCoord2i(0, remoteness ? 0 : 1);
        ::glVertex2f(this->tiles[i].First().Left(), this->tiles[i].First().Top());
        ::glTexCoord2i(1, remoteness ? 0 : 1);
        ::glVertex2f(this->tiles[i].First().Right(), this->tiles[i].First().Top());
        ::glTexCoord2i(1, remoteness ? 1 : 0);
        ::glVertex2f(this->tiles[i].First().Right(), this->tiles[i].First().Bottom());
        ::glEnd();
    }
    ::glBindTexture(GL_TEXTURE_2D, 0);

    ::glDisable(GL_TEXTURE_2D);

    return true;
}


/*
 * imageviewer2::ImageViewer::onFilesPasted
 */
bool imageviewer2::ImageViewer::onFilesPasted(param::ParamSlot& slot) {
    vislib::TString str(this->pasteFilenamesSlot.Param<param::StringParam>()->Value());
    vislib::TString left, right;
    str.Replace(_T("\r"), _T(""));
    this->interpretLine(str, left, right);
    this->leftFilenameSlot.Param<param::FilePathParam>()->SetValue(left);
    this->rightFilenameSlot.Param<param::FilePathParam>()->SetValue(right);
    return true;
}


/*
 * imageviewer2::ImageViewer::interpretLine
 */
void imageviewer2::ImageViewer::interpretLine(
    const vislib::TString source, vislib::TString& left, vislib::TString& right) {
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
}

/*
 * imageviewer2::ImageViewer::onSlideshowPasted
 */
bool imageviewer2::ImageViewer::onSlideshowPasted(param::ParamSlot& slot) {
    vislib::TString left, right;
    this->leftFiles.Clear();
    this->rightFiles.Clear();
    vislib::TString str(this->pasteSlideshowSlot.Param<param::StringParam>()->Value());
    str.Replace(_T("\r"), _T(""));
    vislib::TString::Size startPos = 0;
    vislib::TString::Size pos = str.Find(_T("\n"), startPos);
    while (pos != vislib::TString::INVALID_POS) {
        vislib::TString line = str.Substring(startPos, pos - startPos);
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
bool imageviewer2::ImageViewer::onFirstPressed(param::ParamSlot& slot) {
    this->currentSlot.Param<param::IntParam>()->SetValue(0);
    return true;
}


/*
 * imageviewer2::ImageViewer::onPreviousPressed
 */
bool imageviewer2::ImageViewer::onPreviousPressed(param::ParamSlot& slot) {
    this->currentSlot.Param<param::IntParam>()->SetValue(this->currentSlot.Param<param::IntParam>()->Value() - 1);
    return true;
}


/*
 * imageviewer2::ImageViewer::onNextPressed
 */
bool imageviewer2::ImageViewer::onNextPressed(param::ParamSlot& slot) {
    this->currentSlot.Param<param::IntParam>()->SetValue(this->currentSlot.Param<param::IntParam>()->Value() + 1);
    return true;
}


/*
 * imageviewer2::ImageViewer::onLastPressed
 */
bool imageviewer2::ImageViewer::onLastPressed(param::ParamSlot& slot) {
    this->currentSlot.Param<param::IntParam>()->SetValue(this->leftFiles.Count() - 1);
    return true;
}


/*
 * imageviewer2::ImageViewer::onCurrentSet
 */
bool imageviewer2::ImageViewer::onCurrentSet(param::ParamSlot& slot) {
    int s = slot.Param<param::IntParam>()->Value();
    if (s > -1 && s < this->leftFiles.Count()) {
        this->leftFilenameSlot.Param<param::FilePathParam>()->SetValue(leftFiles[s]);
        this->rightFilenameSlot.Param<param::FilePathParam>()->SetValue(rightFiles[s]);

        // use ResetViewOnBBoxChange of your View!
    }
    return true;
}


/*
 * imageviewer2::ImageViewer::onBlankMachineSet
 */
bool imageviewer2::ImageViewer::onBlankMachineSet(param::ParamSlot& slot) {
    vislib::TString str(this->blankMachine.Param<param::StringParam>()->Value());
    vislib::TString::Size startPos = 0;
    vislib::TString::Size pos = str.Find(_T(";"), startPos);
    blankMachines.Clear();
    while (pos != vislib::TString::INVALID_POS) {
        vislib::TString machine = str.Substring(startPos, pos - startPos);
        machine.ToLowerCase();
        this->blankMachines.Append(machine);
        startPos = pos + 1;
        pos = str.Find(_T(";"), startPos);
    }
    vislib::TString machine = str.Substring(startPos);
    machine.ToLowerCase();
    this->blankMachines.Append(machine);
    return true;
}
