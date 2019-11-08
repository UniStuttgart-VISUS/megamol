/*
 * ImageRenderer.cpp
 *
 * Copyright (C) 2010 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "imageviewer2/ImageRenderer.h"
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
#include "mmcore/view/CallRender3D_2.h"
#include "vislib/sys/Log.h"
#include "vislib/sys/SystemInformation.h"
//#include <cmath>

using namespace megamol::core;
using namespace megamol;

const unsigned int TILE_SIZE = 2 * 1024;

/*
 * misc::ImageRenderer::ImageRenderer
 */
imageviewer2::ImageRenderer::ImageRenderer(void)
    : Renderer3DModule_2()
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
    this->pasteFilenamesSlot.SetUpdateCallback(&ImageRenderer::onFilesPasted);
    this->MakeSlotAvailable(&this->pasteFilenamesSlot);

    this->pasteSlideshowSlot << new param::StringParam("");
    this->pasteSlideshowSlot.SetUpdateCallback(&ImageRenderer::onSlideshowPasted);
    this->MakeSlotAvailable(&this->pasteSlideshowSlot);

    this->firstSlot << new param::ButtonParam();
    this->firstSlot.SetUpdateCallback(&ImageRenderer::onFirstPressed);
    this->MakeSlotAvailable(&this->firstSlot);
    this->previousSlot << new param::ButtonParam(core::view::Key::KEY_PAGE_UP);
    this->previousSlot.SetUpdateCallback(&ImageRenderer::onPreviousPressed);
    this->MakeSlotAvailable(&this->previousSlot);

    this->currentSlot << new param::IntParam(0);
    this->currentSlot.SetUpdateCallback(&ImageRenderer::onCurrentSet);
    this->MakeSlotAvailable(&this->currentSlot);

    this->nextSlot << new param::ButtonParam(core::view::Key::KEY_PAGE_DOWN);
    this->nextSlot.SetUpdateCallback(&ImageRenderer::onNextPressed);
    this->MakeSlotAvailable(&this->nextSlot);
    this->lastSlot << new param::ButtonParam();
    this->lastSlot.SetUpdateCallback(&ImageRenderer::onLastPressed);
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
    this->blankMachine.SetUpdateCallback(&ImageRenderer::onBlankMachineSet);
    this->MakeSlotAvailable(&this->blankMachine);

    vislib::sys::SystemInformation::ComputerName(this->machineName);
    this->machineName.ToLowerCase();

    this->callRequestMpi.SetCompatibleCall<cluster::mpi::MpiCallDescription>();
    this->MakeSlotAvailable(&this->callRequestMpi);

    this->callRequestImage.SetCompatibleCall<image_calls::Image2DCallDescription>();
    this->MakeSlotAvailable(&this->callRequestImage);
}


/*
 * misc::ImageRenderer::~ImageRenderer
 */
imageviewer2::ImageRenderer::~ImageRenderer(void) { this->Release(); }


/*
 * misc::ImageRenderer::create
 */
bool imageviewer2::ImageRenderer::create(void) {
    vislib::graphics::BitmapCodecCollection::DefaultCollection().AddCodec(new sg::graphics::PngBitmapCodec());
    vislib::graphics::BitmapCodecCollection::DefaultCollection().AddCodec(new sg::graphics::JpegBitmapCodec());

    vislib::graphics::gl::ShaderSource vertShader;
    vislib::graphics::gl::ShaderSource fragShader;

    try {
        if (!instance()->ShaderSourceFactory().MakeShaderSource("imageviewer::vertex", vertShader)) {
            return false;
        }
        if (!instance()->ShaderSourceFactory().MakeShaderSource("imageviewer::fragment", fragShader)) {
            return false;
        }
    } catch (vislib::Exception e) {
        vislib::sys::Log::DefaultLog.WriteMsg(
            vislib::sys::Log::LEVEL_ERROR, "[ImageRenderer] Unable to make shader source: %s\n", e.GetMsgA());
    }

    try {
        if (!this->theShader.Create(vertShader.Code(), vertShader.Count(), fragShader.Code(), fragShader.Count())) {
            vislib::sys::Log::DefaultLog.WriteMsg(
                vislib::sys::Log::LEVEL_ERROR, "[ImageRenderer] Unable to compile sphere shader\n");
            return false;
        }
    } catch (vislib::graphics::gl::AbstractOpenGLShader::CompileException ce) {
        vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
            "[SphereRenderer] Unable to compile sphere shader (@%s): %s\n",
            vislib::graphics::gl::AbstractOpenGLShader::CompileException::CompileActionName(ce.FailedAction()),
            ce.GetMsgA());
        return false;
    } catch (vislib::Exception e) {
        vislib::sys::Log::DefaultLog.WriteMsg(
            vislib::sys::Log::LEVEL_ERROR, "[ImageRenderer] Unable to compile shader: %s\n", e.GetMsgA());
        return false;
    } catch (...) {
        vislib::sys::Log::DefaultLog.WriteMsg(
            vislib::sys::Log::LEVEL_ERROR, "[ImageRenderer] Unable to compile shader: Unknown exception\n");
        return false;
    }

    glGenBuffers(1, &theVertBuffer);
    glGenBuffers(1, &theTexCoordBuffer);
    glGenVertexArrays(1, &theVAO);

    return true;
}


/*
 * imageviewer2::ImageRenderer::GetExtents
 */
bool imageviewer2::ImageRenderer::GetExtents(view::CallRender3D_2& call) {

    view::Camera_2 cam;
    call.GetCamera(cam);
    cam_type::snapshot_type snapshot;
    cam_type::matrix_type viewTemp, projTemp;

    // Generate complete snapshot and calculate matrices
    cam.calc_matrices(snapshot, viewTemp, projTemp, thecam::snapshot_content::all);

    auto CamPos = snapshot.position;
    auto CamView = snapshot.view_vector;
    auto CamRight = snapshot.right_vector;
    auto CamUp = snapshot.up_vector;
    auto CamNearClip = snapshot.frustum_near;
    auto Eye = cam.eye();
    bool rightEye = (Eye == core::thecam::Eye::right);

    glm::mat4 view = viewTemp;
    glm::mat4 proj = projTemp;
    auto MVinv = glm::inverse(view);
    auto MVP = proj * view;
    auto MVPinv = glm::inverse(MVP);
    auto MVPtransp = glm::transpose(MVP);

    if (!assertImage(rightEye)) return false;

    call.SetTimeFramesCount(1);
    call.AccessBoundingBoxes().Clear();
    call.AccessBoundingBoxes().SetBoundingBox(
        0.0f, 0.0f, -0.5f, static_cast<float>(this->width), static_cast<float>(this->height), 0.5f);
    call.AccessBoundingBoxes().SetClipBox(call.AccessBoundingBoxes().BoundingBox());

    return true;
}


/*
 * imageviewer2::ImageRenderer::release
 */
void imageviewer2::ImageRenderer::release(void) {
    //    this->image.Release();
    glDeleteBuffers(1, &theVertBuffer);
    glDeleteBuffers(1, &theTexCoordBuffer);
    glDeleteVertexArrays(1, &theVAO);
}


/*
 * imageviewer2::ImageRenderer::assertImage
 */
bool imageviewer2::ImageRenderer::assertImage(bool rightEye) {
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
        vislib::sys::Log::DefaultLog.WriteInfo("ImageRenderer: role %s (rank %i of %i)",
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
    if (filenameSlot->IsDirty() || (imgcConnected /* && imgc->DataHash() != datahash*/) ||
        useMpi) { //< imgc has precedence
        if (!imgcConnected) {
            filenameSlot->ResetDirty();
        }
        const vislib::TString& filename = filenameSlot->Param<param::FilePathParam>()->Value();
        static vislib::graphics::BitmapImage img;
        // static ::sg::graphics::PngBitmapCodec codec;
        // codec.Image() = &img;
        ::glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
        try {
            static bool handIsShaken = false;
#ifdef WITH_MPI
            if (useMpi && !handIsShaken) {
                handIsShaken = true;
                vislib::sys::Log::DefaultLog.WriteInfo("ImageRenderer: IMGC Handshake\n");
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
                    "ImageRenderer: IMGC Handshake result remoteness = %d imgcRank = %d\n", remoteness, roleImgcRank);
            }
#endif /* WITH_MPI */

            if (!beBlank && ((loadedFile != filename) || remoteness)) {
                int fileSize = 0;
                BYTE* allFile = nullptr;
                BYTE* imgc_data_ptr = nullptr;
#ifdef WITH_MPI
                // single node or role boss loads the image
                if (!useMpi || roleRank == roleImgcRank) {
                    vislib::sys::Log::DefaultLog.WriteInfo("ImageRenderer: role %s (rank %i of %i) loads an image",
                        myRole == IMG_BLANK ? "blank" : (myRole == IMG_RIGHT ? "right" : "left"), roleRank, roleSize);
#endif /* WITH_MPI */
                    if (!remoteness) {
                        vislib::sys::Log::DefaultLog.WriteInfo(
                            "ImageRenderer: Loading file '%s' from disk\n", filename.PeekBuffer());
                        vislib::sys::FastFile in;
                        if (in.Open(filename, vislib::sys::File::READ_ONLY, vislib::sys::File::SHARE_READ,
                                vislib::sys::File::OPEN_ONLY)) {
                            fileSize = in.GetSize();
                            allFile = new BYTE[fileSize];
                            in.Read(allFile, fileSize);
                            in.Close();
                        } else {
                            printf("ImageRenderer: failed opening file\n");
                            fileSize = 0;
                        }
                    } else if (roleRank == roleImgcRank) {
                        vislib::sys::Log::DefaultLog.WriteInfo("ImageRenderer: Retrieving image from call\n");
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
                    vislib::sys::Log::DefaultLog.WriteInfo("ImageRenderer: Broadcast root = %d\n", bcastRoot);
                    MPI_Bcast(&fileSize, 1, MPI_INT, bcastRoot, roleComm);
                    vislib::sys::Log::DefaultLog.WriteInfo("ImageRenderer: rank %i of %i (role %s) knows fileSize = %i",
                        roleRank, roleSize, myRole == IMG_BLANK ? "blank" : (myRole == IMG_LEFT ? "left" : "right"),
                        fileSize);
                    if (roleRank != 0) {
                        allFile = new BYTE[fileSize];
                        vislib::sys::Log::DefaultLog.WriteInfo(
                            "ImageRenderer: rank %i of %i (role %s) late allocated file storage", roleRank, roleSize,
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
                    vislib::sys::Log::DefaultLog.WriteInfo("ImageRenderer: Decoding Image\n");
                    if (vislib::graphics::BitmapCodecCollection::DefaultCollection().LoadBitmapImage(
                            img, allFile, fileSize)) {
                        img.Convert(vislib::graphics::BitmapImage::TemplateByteRGB);
                        this->width = img.Width();
                        this->height = img.Height();
                        image_ptr = img.PeekDataAs<BYTE>();
                    } else {
                        printf("ImageRenderer: failed decoding file\n");
                        fileSize = 0;
                        this->width = this->height = 0;
                    }
                } else {
                    vislib::sys::Log::DefaultLog.WriteInfo("ImageRenderer: Decoding IMGC at rank %d\n", roleRank);
                    switch (imgc_enc) {
                    case image_calls::Image2DCall::BMP: {
                        if (vislib::graphics::BitmapCodecCollection::DefaultCollection().LoadBitmapImage(
                                img, allFile, fileSize)) {
                            img.Convert(vislib::graphics::BitmapImage::TemplateByteRGB);
                            this->width = img.Width();
                            this->height = img.Height();
                            image_ptr = img.PeekDataAs<BYTE>();
                        } else {
                            printf("ImageRenderer: failed decoding file\n");
                            fileSize = 0;
                            this->width = this->height = 0;
                        }
                    } break;
                    case image_calls::Image2DCall::SNAPPY: {
                    } break;
                    case image_calls::Image2DCall::RAW:
                    default: {
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
                                this->tiles.Last().Second()->Bind();
                                glGenerateMipmap(GL_TEXTURE_2D);
                                this->tiles.Last().Second()->SetFilter(GL_LINEAR_MIPMAP_LINEAR, GL_LINEAR);
                                this->tiles.Last().Second()->SetWrap(GL_CLAMP_TO_EDGE, GL_CLAMP_TO_EDGE);
                                glBindTexture(GL_TEXTURE_2D, 0);
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
                        "ImageRenderer: rank %i of %i (role %s) entering sync barrier", roleRank, roleSize,
                        myRole == IMG_BLANK ? "blank" : (myRole == IMG_LEFT ? "left" : "right"));
                    MPI_Barrier(roleComm);
                    vislib::sys::Log::DefaultLog.WriteInfo("ImageRenderer: rank %i of %i (role %s) leaving sync barrier",
                        roleRank, roleSize, myRole == IMG_BLANK ? "blank" : (myRole == IMG_LEFT ? "left" : "right"));
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


bool imageviewer2::ImageRenderer::initMPI() {
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
 * imageviewer2::ImageRenderer::Render
 */
bool imageviewer2::ImageRenderer::Render(view::CallRender3D_2& call) {

    view::Camera_2 cam;
    call.GetCamera(cam);
    cam_type::snapshot_type snapshot;
    cam_type::matrix_type viewTemp, projTemp;

    // Generate complete snapshot and calculate matrices
    cam.calc_matrices(snapshot, viewTemp, projTemp, thecam::snapshot_content::all);

    auto CamPos = snapshot.position;
    auto CamView = snapshot.view_vector;
    auto CamRight = snapshot.right_vector;
    auto CamUp = snapshot.up_vector;
    auto CamNearClip = snapshot.frustum_near;
    auto Eye = cam.eye();
    bool rightEye = (Eye == core::thecam::Eye::right);

    glm::mat4 view = viewTemp;
    glm::mat4 proj = projTemp;
    auto MVinv = glm::inverse(view);
    auto MVP = proj * view;
    auto MVPinv = glm::inverse(MVP);
    auto MVPtransp = glm::transpose(MVP);

    static bool buffers_initialized = false;
    if (!buffers_initialized && this->tiles.Count() > 0) {
        std::vector<GLfloat> theTexCoords;
        std::vector<GLfloat> theVertCoords;
        const float halfTexel = (1.0f / TILE_SIZE) * 0.5f;
        const float endTexel = 1.0f - halfTexel;
        const int32_t count = this->tiles.Count() * 4 * 2;
        theTexCoords.reserve(count);
        theVertCoords.reserve(count);
        for (SIZE_T i = 0; i < this->tiles.Count(); i++) {
            theTexCoords.push_back(halfTexel);
            theTexCoords.push_back(remoteness ? halfTexel : endTexel);
            theVertCoords.push_back(this->tiles[i].First().Left());
            theVertCoords.push_back(this->tiles[i].First().Top());
            theTexCoords.push_back(endTexel);
            theTexCoords.push_back(remoteness ? halfTexel : endTexel);
            theVertCoords.push_back(this->tiles[i].First().Right());
            theVertCoords.push_back(this->tiles[i].First().Top());
            theTexCoords.push_back(halfTexel);
            theTexCoords.push_back(remoteness ? endTexel : halfTexel);
            theVertCoords.push_back(this->tiles[i].First().Left());
            theVertCoords.push_back(this->tiles[i].First().Bottom());
            theTexCoords.push_back(endTexel);
            theTexCoords.push_back(remoteness ? endTexel : halfTexel);
            theVertCoords.push_back(this->tiles[i].First().Right());
            theVertCoords.push_back(this->tiles[i].First().Bottom());
        }
        ::glBindVertexArray(this->theVAO);
        ::glBindBuffer(GL_ARRAY_BUFFER, this->theVertBuffer);
        ::glBufferData(GL_ARRAY_BUFFER, theVertCoords.size() * sizeof(GLfloat), theVertCoords.data(), GL_STATIC_DRAW);
        ::glEnableVertexAttribArray(0);
        ::glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, 0);
        ::glBindBuffer(GL_ARRAY_BUFFER, this->theTexCoordBuffer);
        ::glBufferData(GL_ARRAY_BUFFER, theTexCoords.size() * sizeof(GLfloat), theTexCoords.data(), GL_STATIC_DRAW);
        ::glEnableVertexAttribArray(1);
        ::glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 0, 0);
        ::glBindVertexArray(0);
        buffers_initialized = true;
    }

    // param::ParamSlot *filenameSlot = rightEye ? (&this->rightFilenameSlot) : (&this->leftFilenameSlot);
    ::glEnable(GL_TEXTURE_2D);
    if (!assertImage(rightEye)) return false;

    ::glDisable(GL_LINE_SMOOTH);
    ::glDisable(GL_BLEND);
    ::glDisable(GL_LIGHTING);
    ::glEnable(GL_DEPTH_TEST);
    ::glLineWidth(1.0f);
    ::glBindVertexArray(this->theVAO);

    this->theShader.Enable();
    glUniformMatrix4fv(theShader.ParameterLocation("MVP"), 1, GL_FALSE, glm::value_ptr(MVP));
    glActiveTexture(GL_TEXTURE0);
    glUniform1i(theShader.ParameterLocation("texSampler"), 0);

    for (SIZE_T i = 0; i < this->tiles.Count(); i++) {
        this->tiles[i].Second()->Bind();
        ::glDrawArrays(GL_TRIANGLE_STRIP, i * 4, 4);
    }
    this->theShader.Disable();

    ::glBindVertexArray(0);
    ::glBindTexture(GL_TEXTURE_2D, 0);
    ::glDisable(GL_TEXTURE_2D);
    ::glDisable(GL_DEPTH_TEST);

    return true;
}


/*
 * imageviewer2::ImageRenderer::onFilesPasted
 */
bool imageviewer2::ImageRenderer::onFilesPasted(param::ParamSlot& slot) {
    vislib::TString str(this->pasteFilenamesSlot.Param<param::StringParam>()->Value());
    vislib::TString left, right;
    str.Replace(_T("\r"), _T(""));
    this->interpretLine(str, left, right);
    this->leftFilenameSlot.Param<param::FilePathParam>()->SetValue(left);
    this->rightFilenameSlot.Param<param::FilePathParam>()->SetValue(right);
    return true;
}


/*
 * imageviewer2::ImageRenderer::interpretLine
 */
void imageviewer2::ImageRenderer::interpretLine(
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
 * imageviewer2::ImageRenderer::onSlideshowPasted
 */
bool imageviewer2::ImageRenderer::onSlideshowPasted(param::ParamSlot& slot) {
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
 * imageviewer2::ImageRenderer::onFirstPressed
 */
bool imageviewer2::ImageRenderer::onFirstPressed(param::ParamSlot& slot) {
    this->currentSlot.Param<param::IntParam>()->SetValue(0);
    return true;
}


/*
 * imageviewer2::ImageRenderer::onPreviousPressed
 */
bool imageviewer2::ImageRenderer::onPreviousPressed(param::ParamSlot& slot) {
    this->currentSlot.Param<param::IntParam>()->SetValue(this->currentSlot.Param<param::IntParam>()->Value() - 1);
    return true;
}


/*
 * imageviewer2::ImageRenderer::onNextPressed
 */
bool imageviewer2::ImageRenderer::onNextPressed(param::ParamSlot& slot) {
    this->currentSlot.Param<param::IntParam>()->SetValue(this->currentSlot.Param<param::IntParam>()->Value() + 1);
    return true;
}


/*
 * imageviewer2::ImageRenderer::onLastPressed
 */
bool imageviewer2::ImageRenderer::onLastPressed(param::ParamSlot& slot) {
    this->currentSlot.Param<param::IntParam>()->SetValue(this->leftFiles.Count() - 1);
    return true;
}


/*
 * imageviewer2::ImageRenderer::onCurrentSet
 */
bool imageviewer2::ImageRenderer::onCurrentSet(param::ParamSlot& slot) {
    int s = slot.Param<param::IntParam>()->Value();
    if (s > -1 && s < this->leftFiles.Count()) {
        this->leftFilenameSlot.Param<param::FilePathParam>()->SetValue(leftFiles[s]);
        this->rightFilenameSlot.Param<param::FilePathParam>()->SetValue(rightFiles[s]);

        // use ResetViewOnBBoxChange of your View!
    }
    return true;
}


/*
 * imageviewer2::ImageRenderer::onBlankMachineSet
 */
bool imageviewer2::ImageRenderer::onBlankMachineSet(param::ParamSlot& slot) {
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
