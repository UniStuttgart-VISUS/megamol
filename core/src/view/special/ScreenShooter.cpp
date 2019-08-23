/*
 * ScreenShooter.cpp
 *
 * Copyright (C) 2009 - 2010 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "mmcore/view/special/ScreenShooter.h"
#include <climits>
#include <map>
#include <sstream>
#include "mmcore/AbstractNamedObject.h"
#include "mmcore/AbstractNamedObjectContainer.h"
#include "mmcore/CoreInstance.h"
#include "mmcore/param/BoolParam.h"
#include "mmcore/param/ButtonParam.h"
#include "mmcore/param/EnumParam.h"
#include "mmcore/param/FilePathParam.h"
#include "mmcore/param/FloatParam.h"
#include "mmcore/param/IntParam.h"
#include "mmcore/param/StringParam.h"
#include "mmcore/view/CallRenderView.h"
#include "png.h"
#include "vislib/Trace.h"
#include "vislib/assert.h"
#include "vislib/graphics/gl/FramebufferObject.h"
#include "vislib/graphics/gl/IncludeAllGL.h"
#include "vislib/math/mathfunctions.h"
#include "vislib/sys/CriticalSection.h"
#include "vislib/sys/FastFile.h"
#include "vislib/sys/File.h"
#include "vislib/sys/Log.h"
#include "vislib/sys/Thread.h"


namespace megamol {
namespace core {
namespace view {
namespace special {

/**
 * My error handling function for png export
 *
 * @param pngPtr The png structure pointer
 * @param msg The error message
 */
static void PNGAPI myPngError(png_structp pngPtr, png_const_charp msg) {
    throw vislib::Exception(msg, __FILE__, __LINE__);
}

/**
 * My error handling function for png export
 *
 * @param pngPtr The png structure pointer
 * @param msg The error message
 */
static void PNGAPI myPngWarn(png_structp pngPtr, png_const_charp msg) {
    vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_WARN, "Png-Warning: %s\n", msg);
}

/**
 * My write function for png export
 *
 * @param pngPtr The png structure pointer
 * @param buf The pointer to the buffer to be written
 * @param size The number of bytes to be written
 */
static void PNGAPI myPngWrite(png_structp pngPtr, png_bytep buf, png_size_t size) {
    vislib::sys::File* f = static_cast<vislib::sys::File*>(png_get_io_ptr(pngPtr));
    f->Write(buf, size);
}

/**
 * My flush function for png export
 *
 * @param pngPtr The png structure pointer
 */
static void PNGAPI myPngFlush(png_structp pngPtr) {
    vislib::sys::File* f = static_cast<vislib::sys::File*>(png_get_io_ptr(pngPtr));
    f->Flush();
}

/**
 * Data used by the multithreaded shooter code
 */
typedef struct _shooterdata_t {

    /** The two temporary files */
    vislib::sys::File* tmpFiles[2];

    /** The locks for the usage of the temporary files */
    vislib::sys::CriticalSection tmpFileLocks[2];

    /** lock to syncronize the switch of temporary files */
    vislib::sys::CriticalSection switchLock;

    /** The width of the full image */
    unsigned int imgWidth;

    /** The height of the full image */
    unsigned int imgHeight;

    /** The general tile width */
    unsigned int tileWidth;

    /** The general tile height */
    unsigned int tileHeight;

    /** The png export structure */
    png_structp pngPtr;

    /** The png export info */
    png_infop pngInfoPtr;

    /** Bytes per pixel */
    unsigned int bpp;

} ShooterData;

/**
 * The second thread to load the tile data from the temporary files and
 * create the final output
 *
 * @param d Pointer to the common ShooterData structure
 *
 * @return 0
 */
static DWORD myPngStoreData(void* d) {
    ShooterData* data = static_cast<ShooterData*>(d);
    int xSteps = (data->imgWidth / data->tileWidth) + (((data->imgWidth % data->tileWidth) != 0) ? 1 : 0);
    int ySteps = (data->imgHeight / data->tileHeight) + (((data->imgHeight % data->tileHeight) != 0) ? 1 : 0);
    int tmpid = ySteps % 2;
    BYTE* buffer = new BYTE[data->imgWidth * data->bpp]; // 1 scanline at a time

    if (buffer == NULL) {
        vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR, "Unable to allocate scanline buffer");
        return -1;
    }

    data->tmpFileLocks[tmpid].Lock();
    VLTRACE(VISLIB_TRCELVL_INFO, "Writer locked tmp[%d]\n", tmpid);

    for (int yi = ySteps - 1; yi >= 0; yi--) {
        tmpid = yi % 2;
        int tileY = yi * data->tileHeight;
        int tileH = vislib::math::Min(data->tileHeight, data->imgHeight - tileY);
        if (tileH <= 0) {
            return -1;
        }

        data->tmpFileLocks[tmpid].Lock();
        VLTRACE(VISLIB_TRCELVL_INFO, "Writer locked tmp[%d]\n", tmpid);
        VLTRACE(VISLIB_TRCELVL_INFO, "Writer unlocks tmp[%d]\n", (1 - tmpid));
        data->tmpFileLocks[1 - tmpid].Unlock();
        data->switchLock.Lock();
        data->switchLock.Unlock();

        for (int yo = tileH - 1; yo >= 0; yo--) {
            for (int xi = 0; xi < xSteps; xi++) {
                data->tmpFiles[tmpid]->Seek(
                    xi * data->tileWidth * data->tileHeight * data->bpp + yo * data->tileWidth * data->bpp);
                data->tmpFiles[tmpid]->Read(buffer + (xi * data->tileWidth * data->bpp),
                    vislib::math::Min(data->tileWidth, data->imgWidth - xi * data->tileWidth) * data->bpp);
            }
            png_write_row(data->pngPtr, buffer);
        }
    }

    VLTRACE(VISLIB_TRCELVL_INFO, "Writer unlocks tmp[%d]\n", tmpid);
    data->tmpFileLocks[tmpid].Unlock();

    delete[] buffer;

    return 0;
}

} /* end namespace special */
} /* end namespace view */
} /* end namespace core */
} /* end namespace megamol */

using namespace megamol::core;


/*
 * view::special::ScreenShooter::IsAvailable
 */
bool view::special::ScreenShooter::IsAvailable(void) {
    return true;
    // required extensions must be tested lazy,
    //     because open gl can still be missing
}


/*
 * view::special::ScreenShooter::release
 */
view::special::ScreenShooter::ScreenShooter(const bool reducedParameters) : job::AbstractJob(), Module(),
        viewNameSlot("view", "The name of the view instance or view to be used"),
        imgWidthSlot("imgWidth", "The width in pixels of the resulting image"),
        imgHeightSlot("imgHeight", "The height in pixels of the resulting image"),
        tileWidthSlot("tileWidth", "The width of a rendering tile in pixels"),
        tileHeightSlot("tileHeight", "The height of a rendering tile in pixels"),
        imageFilenameSlot("filename", "The file name to store the resulting image under"),
        backgroundSlot("background", "The background to be used"),
        triggerButtonSlot("trigger", "The trigger button"),
        closeAfterShotSlot("closeAfter", "If set the application will close after an image had been created"),
        animFromSlot("anim::from", "The first time"),
        animToSlot("anim::to", "The last time"),
        animStepSlot("anim::step", "The time step"),
        animAddTime2FrameSlot("anim::addTime2Fname", "Add animation time to the output filenames"),
        makeAnimSlot("anim::makeAnim", "Flag whether or not to make an animation of screen shots"),
        animTimeParamNameSlot("anim::paramname", "Name of the time parameter"),
        disableCompressionSlot("disableCompressionSlot", "set compression level to 0"),
        running(false),
        animLastFrameTime(std::numeric_limits<decltype(animLastFrameTime)>::lowest()),
        outputCounter(0) {

    this->viewNameSlot << new param::StringParam("");
    this->MakeSlotAvailable(&this->viewNameSlot);

    this->imgWidthSlot << new param::IntParam(1024, 1);
    this->MakeSlotAvailable(&this->imgWidthSlot);
    this->imgHeightSlot << new param::IntParam(768, 1);
    this->MakeSlotAvailable(&this->imgHeightSlot);

    this->tileWidthSlot << new param::IntParam(1024, 1);
    this->MakeSlotAvailable(&this->tileWidthSlot);
    this->tileHeightSlot << new param::IntParam(1024, 1);
    this->MakeSlotAvailable(&this->tileHeightSlot);

    this->imageFilenameSlot << new param::FilePathParam("Unnamed.png", param::FilePathParam::FLAG_TOBECREATED);
    if (!reducedParameters) this->MakeSlotAvailable(&this->imageFilenameSlot);

    param::EnumParam* bkgnd = new param::EnumParam(0);
    bkgnd->SetTypePair(0, "Scene Background");
    bkgnd->SetTypePair(1, "Transparent");
    bkgnd->SetTypePair(2, "White");
    bkgnd->SetTypePair(3, "Black");
    bkgnd->SetTypePair(4, "Render Raster");
    this->backgroundSlot << bkgnd;
    this->MakeSlotAvailable(&this->backgroundSlot);

    this->triggerButtonSlot << new param::ButtonParam(core::view::Key::KEY_S, core::view::Modifier::ALT);
    this->triggerButtonSlot.SetUpdateCallback(&ScreenShooter::triggerButtonClicked);
    if (!reducedParameters) this->MakeSlotAvailable(&this->triggerButtonSlot);

    this->closeAfterShotSlot << new param::BoolParam(false);
    if (!reducedParameters) this->MakeSlotAvailable(&this->closeAfterShotSlot);

    this->disableCompressionSlot << new param::BoolParam(false);
    if (!reducedParameters) this->MakeSlotAvailable(&this->disableCompressionSlot);

    this->animFromSlot << new param::IntParam(0, 0);
    if (!reducedParameters) this->MakeSlotAvailable(&this->animFromSlot);

    this->animToSlot << new param::IntParam(0, 0);
    if (!reducedParameters) this->MakeSlotAvailable(&this->animToSlot);

    this->animStepSlot << new param::FloatParam(1.0f, 0.01f);
    // this->animStepSlot << new param::IntParam(1, 1);
    if (!reducedParameters) this->MakeSlotAvailable(&this->animStepSlot);

    this->animAddTime2FrameSlot << new param::BoolParam(false);
    if (!reducedParameters) this->MakeSlotAvailable(&this->animAddTime2FrameSlot);

    this->makeAnimSlot << new param::BoolParam(false);
    if (!reducedParameters) this->MakeSlotAvailable(&this->makeAnimSlot);

    this->animTimeParamNameSlot << new param::StringParam("");
    if (!reducedParameters) this->MakeSlotAvailable(&this->animTimeParamNameSlot);
}


/*
 * view::special::ScreenShooter::release
 */
view::special::ScreenShooter::~ScreenShooter() { this->Release(); }


/*
 * view::special::ScreenShooter::release
 */
bool view::special::ScreenShooter::IsRunning(void) const { return this->running; }


/*
 * view::special::ScreenShooter::release
 */
bool view::special::ScreenShooter::Start(void) {
    this->running = true;
    return true;
}


/*
 * view::special::ScreenShooter::release
 */
bool view::special::ScreenShooter::Terminate(void) {
    this->running = false;
    return true;
}


/*
 * view::special::ScreenShooter::release
 */
bool view::special::ScreenShooter::create(void) {
    // Intentionally empty. Initialization is lazy.
    return true;
}


/*
 * view::special::ScreenShooter::release
 */
void view::special::ScreenShooter::release(void) {
    // intentionally empty.
}


/*
 * view::special::ScreenShooter::BeforeRender
 */
void view::special::ScreenShooter::BeforeRender(view::AbstractView* view) {
    using vislib::sys::Log;
    vislib::graphics::gl::FramebufferObject fbo;
    ShooterData data;
    vislib::sys::Thread t2(&myPngStoreData);

    view->UnregisterHook(this); // avoid recursive calling

    data.imgWidth = static_cast<UINT>(vislib::math::Max(0, this->imgWidthSlot.Param<param::IntParam>()->Value()));
    data.imgHeight = static_cast<UINT>(vislib::math::Max(0, this->imgHeightSlot.Param<param::IntParam>()->Value()));
    data.tileWidth = static_cast<UINT>(vislib::math::Max(0, this->tileWidthSlot.Param<param::IntParam>()->Value()));
    data.tileHeight = static_cast<UINT>(vislib::math::Max(0, this->tileHeightSlot.Param<param::IntParam>()->Value()));
    vislib::TString filename = this->imageFilenameSlot.Param<param::FilePathParam>()->Value();
    float frameTime = -1.0f;
    if (this->makeAnimSlot.Param<param::BoolParam>()->Value()) {
        param::ParamSlot* time = this->findTimeParam(view);
        if (time != NULL) {
            frameTime = time->Param<param::FloatParam>()->Value();
            if (frameTime == this->animLastFrameTime) {
                this->makeAnimSlot.Param<param::BoolParam>()->SetValue(false);
                Log::DefaultLog.WriteInfo("Animation screen shooting aborted: time code did not change");
            } else {
                this->animLastFrameTime = frameTime;
                vislib::TString ext;
                ext = filename;
                ext.ToLowerCase();
                if (ext.EndsWith(_T(".png"))) {
                    filename.Truncate(filename.Length() - 4);
                }

                if (this->animAddTime2FrameSlot.Param<param::BoolParam>()->Value()) {
                    int intPart = static_cast<int>(floor(this->animLastFrameTime));
                    float fractPart = this->animLastFrameTime - (float)intPart;
                    ext.Format(_T(".%.5d.%03d.png"), intPart, (int)(fractPart * 1000.0f));
                } else {
                    ext.Format(_T(".%.5u.png"), this->outputCounter);
                }

                outputCounter++;

                filename += ext;
            }
        } else {
            this->makeAnimSlot.Param<param::BoolParam>()->SetValue(false);
            Log::DefaultLog.WriteInfo("Animation screen shooting aborted: unable to fetch time code");
        }
    }
    int bkgndMode = this->backgroundSlot.Param<param::EnumParam>()->Value();
    bool closeAfter = this->closeAfterShotSlot.Param<param::BoolParam>()->Value();
    data.bpp = (bkgndMode == 1) ? 4 : 3;
    data.pngInfoPtr = NULL;
    data.pngPtr = NULL;

    if ((data.tileWidth == 0) || (data.tileHeight == 0)) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "Failed to create Screenshot: Illegal tile size %u x %u",
            data.tileWidth, data.tileHeight);
        return;
    }
    if ((data.imgWidth == 0) || (data.imgHeight == 0)) {
        Log::DefaultLog.WriteMsg(
            Log::LEVEL_ERROR, "Failed to create Screenshot: Illegal image size %u x %u", data.imgWidth, data.imgHeight);
        return;
    }
    if (filename.IsEmpty()) {
        Log::DefaultLog.WriteMsg(
            Log::LEVEL_ERROR, "Failed to create Screenshot: You must specify a file name to save the image");
        return;
    }

    if (!vislib::graphics::gl::FramebufferObject::InitialiseExtensions()) {
        Log::DefaultLog.WriteMsg(
            Log::LEVEL_ERROR, "Failed to create Screenshot: Unable to initialize framebuffer extensions.");
        return;
    }

    data.tmpFiles[0] = vislib::sys::File::CreateTempFile();
    if (data.tmpFiles[0] == NULL) {
        Log::DefaultLog.WriteMsg(
            Log::LEVEL_ERROR, "Failed to create Screenshot: Unable to create first temporary file.");
        return;
    }
    data.tmpFiles[1] = vislib::sys::File::CreateTempFile();
    if (data.tmpFiles[1] == NULL) {
        delete data.tmpFiles[0];
        Log::DefaultLog.WriteMsg(
            Log::LEVEL_ERROR, "Failed to create Screenshot: Unable to create second temporary file.");
        return;
    }

    view::CallRenderView crv;
    BYTE* buffer = NULL;
    vislib::sys::FastFile file;
    bool rollback = false;
    vislib::graphics::gl::FramebufferObject* overlayfbo = NULL;

    try {

        // open final image file
        if (!file.Open(filename, vislib::sys::File::WRITE_ONLY, vislib::sys::File::SHARE_EXCLUSIVE,
                vislib::sys::File::CREATE_OVERWRITE)) {
            throw vislib::Exception("Cannot open output file", __FILE__, __LINE__);
        }

        // init png lib
        data.pngPtr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, &myPngError, &myPngWarn);
        if (data.pngPtr == NULL) {
            throw vislib::Exception("Cannot create png structure", __FILE__, __LINE__);
        }
        data.pngInfoPtr = png_create_info_struct(data.pngPtr);
        if (data.pngInfoPtr == NULL) {
            throw vislib::Exception("Cannot create png info", __FILE__, __LINE__);
        }
        png_set_write_fn(data.pngPtr, static_cast<void*>(&file), &myPngWrite, &myPngFlush);
        png_set_IHDR(data.pngPtr, data.pngInfoPtr, data.imgWidth, data.imgHeight, 8,
            (bkgndMode == 1) ? PNG_COLOR_TYPE_RGB_ALPHA : PNG_COLOR_TYPE_RGB, PNG_INTERLACE_NONE,
            PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);

        if (this->disableCompressionSlot.Param<param::BoolParam>()->Value()) {
            png_set_compression_level(data.pngPtr, 0);
        }

        // todo: just put the whole project file into one string, even better would ofc be
        // to have a legal exif structure (lol)

        // todo: camera settings are not stored without magic knowledge about the view

        std::string serInstances, serModules, serCalls, serParams;
        this->GetCoreInstance()->SerializeGraph(serInstances, serModules, serCalls, serParams);
        auto confstr = serInstances + "\n" + serModules + "\n" + serCalls + "\n" + serParams;
        std::vector<png_byte> tempvec(confstr.begin(), confstr.end());
        tempvec.push_back('\0');
        // auto info = new png_byte[confstr.size()];
        // memcpy(info, confstr.c_str(), confstr.size());
        // png_set_eXIf_1(data.pngPtr, data.pngInfoPtr, sizeof(info), info);
        png_set_eXIf_1(data.pngPtr, data.pngInfoPtr, tempvec.size(), tempvec.data());

        // check how complex the upcoming action is
        if ((data.imgWidth <= data.tileWidth) && (data.imgHeight <= data.tileHeight)) {
            // we can render the whole image in just one call!

            if (!fbo.Create(data.imgWidth, data.imgHeight, GL_RGBA8, GL_RGBA, GL_UNSIGNED_BYTE,
                    vislib::graphics::gl::FramebufferObject::ATTACHMENT_RENDERBUFFER, GL_DEPTH_COMPONENT24)) {
                throw vislib::Exception("Unable to create image framebuffer object.", __FILE__, __LINE__);
            }

            buffer = new BYTE[data.imgWidth * data.imgHeight * data.bpp];
            if (buffer == NULL) {
                throw vislib::Exception("Cannot allocate image buffer.", __FILE__, __LINE__);
            }
            crv.ResetAll();
            switch (bkgndMode) {
            case 0: /* don't set bkgnd */
                break;
            case 1:
                crv.SetBackground(0, 0, 0);
                break;
            case 2:
                crv.SetBackground(255, 255, 255);
                break;
            case 3:
                crv.SetBackground(0, 0, 0);
                break;
            case 4:
                crv.SetBackground(192, 192, 192);
                break;
            default: /* don't set bkgnd */
                break;
            }
            // don't set projection
            if (fbo.Enable() != GL_NO_ERROR) {
                throw vislib::Exception(
                    "Failed to create Screenshot: Cannot enable Framebuffer object", __FILE__, __LINE__);
            }
            glViewport(0, 0, data.imgWidth, data.imgHeight);
            crv.SetOutputBuffer(&fbo, vislib::math::Rectangle<int>(0, 0, data.imgWidth, data.imgHeight));
            crv.SetTile(static_cast<float>(data.imgWidth), static_cast<float>(data.imgHeight), 0.0f, 0.0f,
                static_cast<float>(data.imgWidth), static_cast<float>(data.imgHeight));
            crv.SetTime(frameTime);
            view->OnRenderView(crv); // glClear by SFX
            glFlush();
            fbo.Disable();

            if (fbo.GetColourTexture(buffer, 0, (bkgndMode == 1) ? GL_RGBA : GL_RGB, GL_UNSIGNED_BYTE) != GL_NO_ERROR) {
                throw vislib::Exception("Failed to create Screenshot: Cannot read image data", __FILE__, __LINE__);
            }
            if (bkgndMode == 1) {
                // fixing alpha from premultiplied to postmultiplied
                for (UINT x = 0; x < data.imgWidth; x++) {
                    for (UINT y = 0; y < data.imgHeight; y++) {
                        BYTE* cptr = buffer + 4 * (x + y * data.imgWidth);
                        if (cptr[3] == 0) continue;
                        float r = static_cast<float>(cptr[0]) / 255.0f;
                        float g = static_cast<float>(cptr[1]) / 255.0f;
                        float b = static_cast<float>(cptr[2]) / 255.0f;
                        float a = static_cast<float>(cptr[3]) / 255.0f;
                        r /= a;
                        g /= a;
                        b /= a;
                        cptr[0] = static_cast<BYTE>(vislib::math::Clamp(r * 255.0f, 0.0f, 255.0f));
                        cptr[1] = static_cast<BYTE>(vislib::math::Clamp(g * 255.0f, 0.0f, 255.0f));
                        cptr[2] = static_cast<BYTE>(vislib::math::Clamp(b * 255.0f, 0.0f, 255.0f));
                    }
                }
            }

            BYTE** rows = NULL;
            try {

                rows = new BYTE*[data.imgHeight];
                for (UINT i = 0; i < data.imgHeight; i++) {
                    rows[data.imgHeight - (1 + i)] = buffer + data.bpp * i * data.imgWidth;
                }
                png_set_rows(data.pngPtr, data.pngInfoPtr, rows);

                png_write_png(data.pngPtr, data.pngInfoPtr, PNG_TRANSFORM_IDENTITY, NULL);

                ARY_SAFE_DELETE(rows);

            } catch (...) {
                if (rows != NULL) {
                    ARY_SAFE_DELETE(rows);
                }
                throw;
            }
            // done!

        } else {
            // here we have to render tiles of the image. Woho for optimizing!

            buffer = new BYTE[data.tileWidth * data.tileHeight * data.bpp];
            if (buffer == NULL) {
                throw vislib::Exception("Cannot allocate temporary image buffer", __FILE__, __LINE__);
            }

            if (!fbo.Create(data.tileWidth, data.tileHeight, GL_RGBA8, GL_RGBA, GL_UNSIGNED_BYTE,
                    vislib::graphics::gl::FramebufferObject::ATTACHMENT_RENDERBUFFER, GL_DEPTH_COMPONENT24)) {
                throw vislib::Exception("Unable to create image framebuffer object.", __FILE__, __LINE__);
            }

            int xSteps = (data.imgWidth / data.tileWidth) + (((data.imgWidth % data.tileWidth) != 0) ? 1 : 0);
            int ySteps = (data.imgHeight / data.tileHeight) + (((data.imgHeight % data.tileHeight) != 0) ? 1 : 0);
            if (xSteps * ySteps == 0) {
                throw vislib::Exception("No tiles scheduled for rendering", __FILE__, __LINE__);
            }
            Log::DefaultLog.WriteMsg(Log::LEVEL_INFO + 100, "%d tile(s) scheduled for rendering", xSteps * ySteps);

            // overlay for user information
            overlayfbo = new vislib::graphics::gl::FramebufferObject();
            if (overlayfbo != NULL) {
                int vp[4];
                glGetIntegerv(GL_VIEWPORT, vp);
                if (overlayfbo->Create(vp[0] + vp[2], vp[1] + vp[3], GL_RGBA8, GL_RGBA, GL_UNSIGNED_BYTE,
                        vislib::graphics::gl::FramebufferObject::ATTACHMENT_RENDERBUFFER, GL_DEPTH_COMPONENT24)) {
                    if (overlayfbo->Enable() == GL_NO_ERROR) {
                        crv.ResetAll();
                        crv.SetTime(frameTime);
                        view->OnRenderView(crv); // glClear by SFX
                        glFlush();
                        overlayfbo->Disable();
                    } else {
                        SAFE_DELETE(overlayfbo);
                    }
                } else {
                    SAFE_DELETE(overlayfbo);
                }
            }

            // start writing
            png_write_info(data.pngPtr, data.pngInfoPtr);

            glDrawBuffer(GL_FRONT);
            glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
            glFlush();
            glDrawBuffer(GL_BACK);

            // render tiles
            for (int yi = ySteps - 1; yi >= 0; yi--) {
                int tmpid = yi % 2;

                data.switchLock.Lock();
                data.tmpFileLocks[tmpid].Lock();
                VLTRACE(VISLIB_TRCELVL_INFO, "Renderer locked tmp[%d]\n", tmpid);
                data.switchLock.Unlock();

                if (yi == ySteps - 1) {
                    // start thread when the first line gets drawn
                    t2.Start(&data);
                }

                int tileY = yi * data.tileHeight;
                int tileH = vislib::math::Min(data.tileHeight, data.imgHeight - tileY);
                int xid = (yi % 2) * 2 - 1; // for the coolness!
                for (int xi = (xid > 0) ? 0 : (xSteps - 1); ((xid > 0) && (xi < xSteps)) || ((xid < 0) && (xi >= 0));
                     xi += xid) {
                    int tileX = xi * data.tileWidth;
                    int tileW = vislib::math::Min(data.tileWidth, data.imgWidth - tileX);

                    if (overlayfbo != NULL) {
                        float tx, ty, tw, th;

                        tx = static_cast<float>(tileX) * static_cast<float>(overlayfbo->GetWidth()) /
                             static_cast<float>(data.imgWidth);
                        ty = static_cast<float>(tileY) * static_cast<float>(overlayfbo->GetHeight()) /
                             static_cast<float>(data.imgHeight);
                        tw = static_cast<float>(tileW) * static_cast<float>(overlayfbo->GetWidth()) /
                             static_cast<float>(data.imgWidth);
                        th = static_cast<float>(tileH) * static_cast<float>(overlayfbo->GetHeight()) /
                             static_cast<float>(data.imgHeight);

                        glDrawBuffer(GL_FRONT);
                        glMatrixMode(GL_PROJECTION);
                        glLoadIdentity();
                        glTranslatef(-1.0f, -1.0f, 0.0f);
                        glScalef(2.0f / static_cast<float>(overlayfbo->GetWidth()),
                            2.0f / static_cast<float>(overlayfbo->GetHeight()), 1.0f);
                        glMatrixMode(GL_MODELVIEW);
                        glLoadIdentity();
                        glDisable(GL_LIGHTING);
                        glDisable(GL_DEPTH_TEST);
                        glEnable(GL_BLEND);
                        glDisable(GL_TEXTURE_2D);
                        glBlendFunc(GL_SRC_ALPHA, GL_ONE);
                        glEnable(GL_LINE_WIDTH);
                        glLineWidth(1.5f);
                        glColor3ub(255, 255, 255);

                        glBegin(GL_LINE_STRIP);
                        glVertex2f(tx + tw * 0.2f, ty + th * 0.01f);
                        glVertex2f(tx + tw * 0.02f, ty + th * 0.01f);
                        glVertex2f(tx + tw * 0.01f, ty + th * 0.02f);
                        glVertex2f(tx + tw * 0.01f, ty + th * 0.2f);
                        glEnd();
                        glBegin(GL_LINE_STRIP);
                        glVertex2f(tx + tw * 0.8f, ty + th * 0.01f);
                        glVertex2f(tx + tw * 0.98f, ty + th * 0.01f);
                        glVertex2f(tx + tw * 0.99f, ty + th * 0.02f);
                        glVertex2f(tx + tw * 0.99f, ty + th * 0.2f);
                        glEnd();
                        glBegin(GL_LINE_STRIP);
                        glVertex2f(tx + tw * 0.2f, ty + th * 0.99f);
                        glVertex2f(tx + tw * 0.02f, ty + th * 0.99f);
                        glVertex2f(tx + tw * 0.01f, ty + th * 0.98f);
                        glVertex2f(tx + tw * 0.01f, ty + th * 0.8f);
                        glEnd();
                        glBegin(GL_LINE_STRIP);
                        glVertex2f(tx + tw * 0.8f, ty + th * 0.99f);
                        glVertex2f(tx + tw * 0.98f, ty + th * 0.99f);
                        glVertex2f(tx + tw * 0.99f, ty + th * 0.98f);
                        glVertex2f(tx + tw * 0.99f, ty + th * 0.8f);
                        glEnd();

                        glFlush();
                        glDrawBuffer(GL_BACK);
                    }

                    // render a tile
                    crv.ResetAll();
                    switch (bkgndMode) {
                    case 0: /* don't set bkgnd */
                        break;
                    case 1:
                        crv.SetBackground(0, 0, 0);
                        break;
                    case 2:
                        crv.SetBackground(255, 255, 255);
                        break;
                    case 3:
                        crv.SetBackground(0, 0, 0);
                        break;
                    case 4:
                        if ((xi + yi) % 2) {
                            crv.SetBackground(192, 192, 192);
                        } else {
                            crv.SetBackground(128, 128, 128);
                        }
                        break;
                    default: /* don't set bkgnd */
                        break;
                    }
                    // don't set projection
                    if (fbo.Enable() != GL_NO_ERROR) {
                        throw vislib::Exception(
                            "Failed to create Screenshot: Cannot enable Framebuffer object", __FILE__, __LINE__);
                    }
                    glViewport(0, 0, tileW, tileH);
                    crv.SetOutputBuffer(&fbo, vislib::math::Rectangle<int>(0, 0, tileW, tileH));
                    crv.SetTile(static_cast<float>(data.imgWidth), static_cast<float>(data.imgHeight),
                        static_cast<float>(tileX), static_cast<float>(tileY), static_cast<float>(tileW),
                        static_cast<float>(tileH));
                    crv.SetTime(frameTime);
                    view->OnRenderView(crv); // glClear by SFX
                    glFlush();
                    fbo.Disable();

                    if (fbo.GetColourTexture(buffer, 0, (bkgndMode == 1) ? GL_RGBA : GL_RGB, GL_UNSIGNED_BYTE) !=
                        GL_NO_ERROR) {
                        throw vislib::Exception(
                            "Failed to create Screenshot: Cannot read image data", __FILE__, __LINE__);
                    }
                    if (bkgndMode == 1) {
                        // fixing alpha from premultiplied to postmultiplied
                        for (UINT x = 0; x < data.tileWidth; x++) {
                            for (UINT y = 0; y < data.tileHeight; y++) {
                                BYTE* cptr = buffer + 4 * (x + y * data.tileWidth);
                                if (cptr[3] == 0) continue;
                                float r = static_cast<float>(cptr[0]) / 255.0f;
                                float g = static_cast<float>(cptr[1]) / 255.0f;
                                float b = static_cast<float>(cptr[2]) / 255.0f;
                                float a = static_cast<float>(cptr[3]) / 255.0f;
                                r /= a;
                                g /= a;
                                b /= a;
                                cptr[0] = static_cast<BYTE>(vislib::math::Clamp(r * 255.0f, 0.0f, 255.0f));
                                cptr[1] = static_cast<BYTE>(vislib::math::Clamp(g * 255.0f, 0.0f, 255.0f));
                                cptr[2] = static_cast<BYTE>(vislib::math::Clamp(b * 255.0f, 0.0f, 255.0f));
                            }
                        }
                    }

                    data.tmpFiles[tmpid]->Seek(xi * data.tileWidth * data.tileHeight * data.bpp);
                    data.tmpFiles[tmpid]->Write(buffer, data.tileWidth * data.tileHeight * data.bpp);

                    if (overlayfbo != NULL) {
                        float tx, ty, tw, th;

                        tx = static_cast<float>(tileX) * static_cast<float>(overlayfbo->GetWidth()) /
                             static_cast<float>(data.imgWidth);
                        ty = static_cast<float>(tileY) * static_cast<float>(overlayfbo->GetHeight()) /
                             static_cast<float>(data.imgHeight);
                        tw = static_cast<float>(tileW) * static_cast<float>(overlayfbo->GetWidth()) /
                             static_cast<float>(data.imgWidth);
                        th = static_cast<float>(tileH) * static_cast<float>(overlayfbo->GetHeight()) /
                             static_cast<float>(data.imgHeight);
                        tx -= 1.0f;
                        ty -= 1.0f;
                        tw += 2.0f;
                        th += 2.0f;

                        glDrawBuffer(GL_FRONT);
                        glMatrixMode(GL_PROJECTION);
                        glLoadIdentity();
                        glTranslatef(-1.0f, -1.0f, 0.0f);
                        glScalef(2.0f / static_cast<float>(overlayfbo->GetWidth()),
                            2.0f / static_cast<float>(overlayfbo->GetHeight()), 1.0f);
                        glMatrixMode(GL_MODELVIEW);
                        glLoadIdentity();
                        glDisable(GL_LIGHTING);
                        glDisable(GL_DEPTH_TEST);
                        glDisable(GL_BLEND);
                        glEnable(GL_TEXTURE_2D);
                        glDisable(GL_CULL_FACE);
                        overlayfbo->BindColourTexture();

                        glBegin(GL_QUADS);
                        glTexCoord2f(tx / static_cast<float>(overlayfbo->GetWidth()),
                            ty / static_cast<float>(overlayfbo->GetHeight()));
                        glVertex2f(tx, ty);
                        glTexCoord2f(tx / static_cast<float>(overlayfbo->GetWidth()),
                            (ty + th) / static_cast<float>(overlayfbo->GetHeight()));
                        glVertex2f(tx, ty + th);
                        glTexCoord2f((tx + tw) / static_cast<float>(overlayfbo->GetWidth()),
                            (ty + th) / static_cast<float>(overlayfbo->GetHeight()));
                        glVertex2f(tx + tw, ty + th);
                        glTexCoord2f((tx + tw) / static_cast<float>(overlayfbo->GetWidth()),
                            ty / static_cast<float>(overlayfbo->GetHeight()));
                        glVertex2f(tx + tw, ty);
                        glEnd();

                        glDrawBuffer(GL_BACK);
                    } /* end if */

                } /* end for xi */

                VLTRACE(VISLIB_TRCELVL_INFO, "Renderer unlocks tmp[%d]\n", tmpid);
                data.tmpFileLocks[tmpid].Unlock();

            } /* end for yi */

            t2.Join();

            png_write_end(data.pngPtr, data.pngInfoPtr);

        } /* end if */

    } catch (vislib::Exception ex) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "Failed to create screenshot image: %s (%s, %d)", ex.GetMsgA(),
            ex.GetFile(), ex.GetLine());
        rollback = true;
    } catch (...) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "Failed to create screenshot image: Unexpected exception");
        rollback = true;
    }

    if (t2.IsRunning()) {
        data.imgHeight = 0;
        t2.Join();
    }
    if (overlayfbo != NULL) {
        try {
            overlayfbo->Release();
        } catch (...) {
        }
        delete overlayfbo;
    }
    if (data.pngPtr != NULL) {
        if (data.pngInfoPtr != NULL) {
            png_destroy_write_struct(&data.pngPtr, &data.pngInfoPtr);
        } else {
            png_destroy_write_struct(&data.pngPtr, (png_infopp)NULL);
        }
    }
    try {
        file.Flush();
    } catch (...) {
    }
    try {
        file.Close();
    } catch (...) {
    }
    if (rollback) {
        try {
            if (vislib::sys::File::Exists(filename)) {
                vislib::sys::File::Delete(filename);
            }
        } catch (...) {
        }
    }
    if (data.tmpFiles[0] != NULL) {
        try {
            data.tmpFiles[0]->Close();
        } catch (...) {
        }
        delete data.tmpFiles[0];
    }
    if (data.tmpFiles[1] != NULL) {
        try {
            data.tmpFiles[1]->Close();
        } catch (...) {
        }
        delete data.tmpFiles[1];
    }
    delete[] buffer;
    fbo.Release();

    vislib::sys::Log::DefaultLog.WriteInfo("Screen shot stored");

    if (this->makeAnimSlot.Param<param::BoolParam>()->Value()) {
        if (this->animLastFrameTime >= this->animToSlot.Param<param::IntParam>()->Value()) {
            Log::DefaultLog.WriteInfo("Animation screen shots complete");

            // stop animation
            // param::ParamSlot *playSlot = dynamic_cast<param::ParamSlot*>(view->FindNamedObject("anim::play"));
            // if (playSlot != NULL)
            //    playSlot->Param<param::BoolParam>()->SetValue(false);
            this->outputCounter = 0;
        } else {
            param::ParamSlot* time = this->findTimeParam(view);
            if (time != NULL) {
                float nextTime = this->animLastFrameTime + this->animStepSlot.Param<param::FloatParam>()->Value();
                time->Param<param::FloatParam>()->SetValue(static_cast<float>(nextTime));
                closeAfter = false;

                view->RegisterHook(this); // ready for the next frame

            } else {
                this->makeAnimSlot.Param<param::BoolParam>()->SetValue(false);
                Log::DefaultLog.WriteInfo("Animation screen shooting aborted: unable to fetch time code");
            }
        }
    }

    if (closeAfter) {
        this->running = false;
        this->GetCoreInstance()->Shutdown();
    }
}


/*
 * view::special::ScreenShooter::createScreenshot
 */
void view::special::ScreenShooter::createScreenshot(const std::string& filename) {
    this->imageFilenameSlot.Param<param::FilePathParam>()->SetValue(filename.c_str());

    triggerButtonClicked(this->triggerButtonSlot);
}


/*
 * view::special::ScreenShooter::triggerButtonClicked
 */
bool view::special::ScreenShooter::triggerButtonClicked(param::ParamSlot& slot) {
    // happy trigger finger hit button action happend
    using vislib::sys::Log;
    ASSERT(&slot == &this->triggerButtonSlot);

    vislib::StringA mvn(this->viewNameSlot.Param<param::StringParam>()->Value());
    Log::DefaultLog.WriteMsg(Log::LEVEL_INFO + 100, "ScreenShot of \"%s\" requested", mvn.PeekBuffer());

    vislib::sys::AutoLock lock(this->ModuleGraphLock());
    {
        AbstractNamedObjectContainer::ptr_type anoc =
            AbstractNamedObjectContainer::dynamic_pointer_cast(this->RootModule());
        AbstractNamedObject::ptr_type ano = anoc->FindChild(mvn);
        ViewInstance* vi = dynamic_cast<ViewInstance*>(ano.get());
        auto av = dynamic_cast<AbstractView*>(ano.get());
        if (vi != nullptr) {
            if (vi->View() != nullptr) {
                av = vi->View();
            }
        }
        if (av != nullptr) {
            if (this->makeAnimSlot.Param<param::BoolParam>()->Value()) {
                param::ParamSlot* timeSlot = this->findTimeParam(vi->View());
                if (timeSlot != nullptr) {
                    timeSlot->Param<param::FloatParam>()->SetValue(
                        static_cast<float>(this->animFromSlot.Param<param::IntParam>()->Value()));
                    this->animLastFrameTime = (float)UINT_MAX;
                } else {
                    Log::DefaultLog.WriteError("Unable to make animation screen shots");
                    this->makeAnimSlot.Param<param::BoolParam>()->SetValue(false);
                }
                // this is not a good idea because the animation module interferes with the "anim::time" parameter in
                // "play" mode ...
                // param::ParamSlot *playSlot =
                // dynamic_cast<param::ParamSlot*>(vi->View()->FindNamedObject("anim::play")); if (playSlot != NULL)
                //    playSlot->Param<param::BoolParam>()->SetValue(true);
            }
            av->RegisterHook(this);
        } else {
            // suppose a view was actually intended!
            bool found = false;
            const auto fun = [this, &found](AbstractView* v) {
                v->RegisterHook(this);
                found = true;
            };
            this->GetCoreInstance()->FindModuleNoLock<AbstractView>(mvn.PeekBuffer(), fun);
            if (!found) {
                if (vi == nullptr) {
                    Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
                        "Unable to find view or viewInstance \"%s\" for ScreenShot", mvn.PeekBuffer());
                } else if (av == nullptr) {
                    Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
                        "ViewInstance \"%s\" is not usable for ScreenShot (Not initialized) and AbstractView \"%s\" "
                        "does not exist either",
                        vi->FullName().PeekBuffer(), mvn.PeekBuffer());
                }
            }
        }
    }

    return true;
}


/*
 * view::special::ScreenShooter::findTimeParam
 */
param::ParamSlot* view::special::ScreenShooter::findTimeParam(view::AbstractView* view) {
    vislib::TString name(this->animTimeParamNameSlot.Param<param::StringParam>()->Value());
    param::ParamSlot* timeSlot = nullptr;

    if (name.IsEmpty()) {
        timeSlot = dynamic_cast<param::ParamSlot*>(view->FindNamedObject("anim::time").get());
    } else {
        AbstractNamedObjectContainer* anoc = dynamic_cast<AbstractNamedObjectContainer*>(view->RootModule().get());
        timeSlot = dynamic_cast<param::ParamSlot*>(anoc->FindNamedObject(vislib::StringA(name)).get());
    }

    return timeSlot;
}
