/*
 * ScreenShooter.cpp
 *
 * Copyright (C) 2009 - 2010 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#include "mmcore_gl/view/special/ScreenShooter.h"

#include <climits>
#include <limits>
#include <map>
#include <sstream>

#include "mmcore/AbstractNamedObject.h"
#include "mmcore/AbstractNamedObjectContainer.h"
#include "mmcore/CoreInstance.h"
#include "mmcore/MegaMolGraph.h"
#include "mmcore/param/BoolParam.h"
#include "mmcore/param/ButtonParam.h"
#include "mmcore/param/EnumParam.h"
#include "mmcore/param/FilePathParam.h"
#include "mmcore/param/FloatParam.h"
#include "mmcore/param/IntParam.h"
#include "mmcore/param/StringParam.h"
#include "mmcore/utility/DateTime.h"
#include "mmcore/utility/graphics/ScreenShotComments.h"
#include "mmcore/utility/log/Log.h"
#include "mmcore_gl/view/CallRenderViewGL.h"
#include "png.h"
#include "vislib/Trace.h"
#include "vislib/assert.h"
#include "vislib/math/mathfunctions.h"
#include "vislib/sys/CriticalSection.h"
#include "vislib/sys/FastFile.h"
#include "vislib/sys/File.h"
#include "vislib/sys/Thread.h"
#include "vislib_gl/graphics/gl/IncludeAllGL.h"

namespace megamol {
namespace core_gl {
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
    megamol::core::utility::log::Log::DefaultLog.WriteMsg(
        megamol::core::utility::log::Log::LEVEL_WARN, "Png-Warning: %s\n", msg);
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
        megamol::core::utility::log::Log::DefaultLog.WriteMsg(
            megamol::core::utility::log::Log::LEVEL_ERROR, "Unable to allocate scanline buffer");
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
} // namespace core_gl
} /* end namespace megamol */

using namespace megamol::core_gl;


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
view::special::ScreenShooter::ScreenShooter(const bool reducedParameters)
        : core::job::AbstractJob()
        , Module()
        , viewNameSlot("view", "The name of the view instance or view to be used")
        , imgWidthSlot("imgWidth", "The width in pixels of the resulting image")
        , imgHeightSlot("imgHeight", "The height in pixels of the resulting image")
        , tileWidthSlot("tileWidth", "The width of a rendering tile in pixels")
        , tileHeightSlot("tileHeight", "The height of a rendering tile in pixels")
        , imageFilenameSlot("filename", "The file name to store the resulting image under")
        , backgroundSlot("background", "The background to be used")
        , triggerButtonSlot("trigger", "The trigger button")
        , closeAfterShotSlot("closeAfter", "If set the application will close after an image had been created")
        , animFromSlot("anim::from", "The first time")
        , animToSlot("anim::to", "The last time")
        , animStepSlot("anim::step", "The time step")
        , animAddTime2FrameSlot("anim::addTime2Fname", "Add animation time to the output filenames")
        , makeAnimSlot("anim::makeAnim", "Flag whether or not to make an animation of screen shots")
        , animTimeParamNameSlot("anim::paramname", "Name of the time parameter. Default if blank: 'anim::time'")
        , disableCompressionSlot("disableCompressionSlot", "set compression level to 0")
        , running(false)
        , animLastFrameTime(std::numeric_limits<decltype(animLastFrameTime)>::lowest())
        , outputCounter(0)
        , currentFbo(nullptr) {

    this->viewNameSlot << new core::param::StringParam("");
    this->MakeSlotAvailable(&this->viewNameSlot);

    this->imgWidthSlot << new core::param::IntParam(1920, 1);
    this->MakeSlotAvailable(&this->imgWidthSlot);
    this->imgHeightSlot << new core::param::IntParam(1080, 1);
    this->MakeSlotAvailable(&this->imgHeightSlot);

    this->tileWidthSlot << new core::param::IntParam(1920, 1);
    this->MakeSlotAvailable(&this->tileWidthSlot);
    this->tileHeightSlot << new core::param::IntParam(1080, 1);
    this->MakeSlotAvailable(&this->tileHeightSlot);

    this->imageFilenameSlot << new core::param::FilePathParam(
        "Unnamed.png", core::param::FilePathParam::Flag_File_ToBeCreatedWithRestrExts, {"png"});
    if (!reducedParameters)
        this->MakeSlotAvailable(&this->imageFilenameSlot);

    core::param::EnumParam* background = new core::param::EnumParam(0);
    background->SetTypePair(0, "Scene Background");
    background->SetTypePair(1, "Transparent");
    background->SetTypePair(2, "White");
    background->SetTypePair(3, "Black");
    background->SetTypePair(4, "Render Raster");
    this->backgroundSlot << background;
    this->MakeSlotAvailable(&this->backgroundSlot);

    this->triggerButtonSlot << new core::param::ButtonParam(core::view::Key::KEY_S, core::view::Modifier::ALT);
    this->triggerButtonSlot.SetUpdateCallback(&ScreenShooter::triggerButtonClicked);
    if (!reducedParameters)
        this->MakeSlotAvailable(&this->triggerButtonSlot);

    this->closeAfterShotSlot << new core::param::BoolParam(false);
    if (!reducedParameters)
        this->MakeSlotAvailable(&this->closeAfterShotSlot);

    this->disableCompressionSlot << new core::param::BoolParam(false);
    if (!reducedParameters)
        this->MakeSlotAvailable(&this->disableCompressionSlot);

    this->animFromSlot << new core::param::IntParam(0, 0);
    if (!reducedParameters)
        this->MakeSlotAvailable(&this->animFromSlot);

    this->animToSlot << new core::param::IntParam(0, 0);
    if (!reducedParameters)
        this->MakeSlotAvailable(&this->animToSlot);

    this->animStepSlot << new core::param::FloatParam(1.0f, 0.01f);
    // this->animStepSlot << new core::param::IntParam(1, 1);
    if (!reducedParameters)
        this->MakeSlotAvailable(&this->animStepSlot);

    this->animAddTime2FrameSlot << new core::param::BoolParam(false);
    if (!reducedParameters)
        this->MakeSlotAvailable(&this->animAddTime2FrameSlot);

    this->makeAnimSlot << new core::param::BoolParam(false);
    if (!reducedParameters)
        this->MakeSlotAvailable(&this->makeAnimSlot);

    this->animTimeParamNameSlot << new core::param::StringParam("");
    if (!reducedParameters)
        this->MakeSlotAvailable(&this->animTimeParamNameSlot);

    /// XXX Disable tiling option since it is not working for new megamol frontend (yet)
    this->tileWidthSlot.Parameter()->SetGUIVisible(false);
    this->tileHeightSlot.Parameter()->SetGUIVisible(false);
    this->closeAfterShotSlot.Parameter()->SetGUIVisible(false);
}


/*
 * view::special::ScreenShooter::release
 */
view::special::ScreenShooter::~ScreenShooter() {
    this->Release();
}


/*
 * view::special::ScreenShooter::release
 */
bool view::special::ScreenShooter::IsRunning(void) const {
    return this->running;
}


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
    currentFbo = std::make_shared<glowl::FramebufferObject>(1, 1);
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
void view::special::ScreenShooter::BeforeRender(core::view::AbstractView* view) {
    using megamol::core::utility::log::Log;
    ShooterData data;
    vislib::sys::Thread t2(&myPngStoreData);

    view->UnregisterHook(this); // avoid recursive calling

    data.imgWidth = static_cast<UINT>(vislib::math::Max(0, this->imgWidthSlot.Param<core::param::IntParam>()->Value()));
    data.imgHeight =
        static_cast<UINT>(vislib::math::Max(0, this->imgHeightSlot.Param<core::param::IntParam>()->Value()));
    data.tileWidth =
        static_cast<UINT>(vislib::math::Max(0, this->tileWidthSlot.Param<core::param::IntParam>()->Value()));
    data.tileHeight =
        static_cast<UINT>(vislib::math::Max(0, this->tileHeightSlot.Param<core::param::IntParam>()->Value()));

    /// XXX Disable tiling option since it is not working for new megamol frontend (yet)
    data.tileWidth = data.imgWidth;
    data.tileHeight = data.imgHeight;

    vislib::TString filename =
        this->imageFilenameSlot.Param<core::param::FilePathParam>()->Value().generic_u8string().c_str();
    float frameTime = -1.0f;
    if (this->makeAnimSlot.Param<core::param::BoolParam>()->Value()) {
        core::param::ParamSlot* time = this->findTimeParam(view);
        if (time != NULL) {
            frameTime = time->Param<core::param::FloatParam>()->Value();
            if (frameTime == this->animLastFrameTime) {
                this->makeAnimSlot.Param<core::param::BoolParam>()->SetValue(false);
                Log::DefaultLog.WriteInfo("Animation screen shooting aborted: time code did not change");
            } else {
                this->animLastFrameTime = frameTime;
                vislib::TString ext;
                ext = filename;
                ext.ToLowerCase();
                if (ext.EndsWith(_T(".png"))) {
                    filename.Truncate(filename.Length() - 4);
                }

                if (this->animAddTime2FrameSlot.Param<core::param::BoolParam>()->Value()) {
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
            this->makeAnimSlot.Param<core::param::BoolParam>()->SetValue(false);
            Log::DefaultLog.WriteInfo("Animation screen shooting aborted: unable to fetch time code");
        }
    }
    int backgroundMode = this->backgroundSlot.Param<core::param::EnumParam>()->Value();
    bool closeAfter = this->closeAfterShotSlot.Param<core::param::BoolParam>()->Value();
    data.bpp = (backgroundMode == 1) ? 4 : 3;
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

    view::CallRenderViewGL crv;
    BYTE* buffer = NULL;
    vislib::sys::FastFile file;
    bool rollback = false;
    std::shared_ptr<glowl::FramebufferObject> overlayfbo;
    float bckcolalpha = 1.0f;

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

        if (this->disableCompressionSlot.Param<core::param::BoolParam>()->Value()) {
            png_set_compression_level(data.pngPtr, 0);
        }

        // todo: just put the whole project file into one string, even better would ofc be
        // to have a legal exif structure (lol)

        // todo: camera settings are not stored without magic knowledge about the view
        megamol::core::utility::graphics::ScreenShotComments ssc(this->GetCoreInstance()->SerializeGraph());

        png_set_text(data.pngPtr, data.pngInfoPtr, ssc.GetComments().data(), ssc.GetComments().size());

        png_set_IHDR(data.pngPtr, data.pngInfoPtr, data.imgWidth, data.imgHeight, 8,
            (backgroundMode == 1) ? PNG_COLOR_TYPE_RGB_ALPHA : PNG_COLOR_TYPE_RGB, PNG_INTERLACE_NONE,
            PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);

        // check how complex the upcoming action is
        if ((data.imgWidth <= data.tileWidth) && (data.imgHeight <= data.tileHeight)) {
            // we can render the whole image in just one call!

            glBindFramebuffer(GL_FRAMEBUFFER, 0); // better safe then sorry, "unbind" fbo before delting one
            try {
                currentFbo = std::make_shared<glowl::FramebufferObject>(
                    data.imgWidth, data.imgHeight, glowl::FramebufferObject::DEPTH24);
                currentFbo->createColorAttachment(GL_RGBA8, GL_RGBA, GL_UNSIGNED_BYTE);

                // TODO: check completness and throw if not?
            } catch (glowl::FramebufferObjectException const& exc) {
                megamol::core::utility::log::Log::DefaultLog.WriteError(
                    "[ScreenShooter] Unable to create framebuffer object: %s\n", exc.what());
            }

            buffer = new BYTE[data.imgWidth * data.imgHeight * data.bpp];
            if (buffer == NULL) {
                throw vislib::Exception("Cannot allocate image buffer.", __FILE__, __LINE__);
            }

            switch (backgroundMode) {
            case 0:
                crv.SetBackgroundColor(view->BackgroundColor());
                break;
            case 1:
                crv.SetBackgroundColor(glm::vec4(0, 0, 0, 1));
                break;
            case 2:
                crv.SetBackgroundColor(glm::vec4(1, 1, 1, 1));
                break;
            case 3:
                crv.SetBackgroundColor(glm::vec4(0, 0, 0, 1));
                break;
            case 4:
                crv.SetBackgroundColor(glm::vec4(255.0f / 192.0f, 255.0f / 192.0f, 255.0f / 192.0f, 1));
                break;
            default: /* don't set background */
                break;
            }

            crv.SetFramebuffer(currentFbo);
            crv.SetTime(frameTime);
            //crv.SetTile(static_cast<float>(data.imgWidth), static_cast<float>(data.imgHeight), 0.0f, 0.0f,
            //    static_cast<float>(data.imgWidth), static_cast<float>(data.imgHeight));
            view->OnRenderView(crv); // glClear by SFX
            //view->Resize(static_cast<unsigned int>(vp[2]), static_cast<unsigned int>(vp[3]));
            glFlush();

            currentFbo->bindToRead(0);
            glGetError();
            glReadPixels(0, 0, currentFbo->getWidth(), currentFbo->getHeight(),
                (backgroundMode == 1) ? GL_RGBA : GL_RGB, GL_UNSIGNED_BYTE, buffer);
            glBindFramebuffer(GL_READ_FRAMEBUFFER, 0);
            if (glGetError() != GL_NO_ERROR) {
                megamol::core::utility::log::Log::DefaultLog.WriteError(
                    "[ScreenShooter] Failed to create Screenshot: Cannot read image data.\n");
            }

            if (backgroundMode == 1) {
                // fixing alpha from premultiplied to postmultiplied
                for (UINT x = 0; x < data.imgWidth; x++) {
                    for (UINT y = 0; y < data.imgHeight; y++) {
                        BYTE* cptr = buffer + 4 * (x + y * data.imgWidth);
                        if (cptr[3] == 0)
                            continue;
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

            /// XXX Tiling currently breaks for unknown reason ...
            throw vislib::Exception("[ScreenShooter] Tiling is currently not supported.", __FILE__, __LINE__);

            buffer = new BYTE[data.tileWidth * data.tileHeight * data.bpp];
            if (buffer == NULL) {
                throw vislib::Exception("Cannot allocate temporary image buffer", __FILE__, __LINE__);
            }

            glBindFramebuffer(GL_FRAMEBUFFER, 0); // better safe then sorry, "unbind" fbo before delting one
            try {
                currentFbo = std::make_shared<glowl::FramebufferObject>(
                    data.tileWidth, data.tileHeight, glowl::FramebufferObject::DEPTH24);
                currentFbo->createColorAttachment(GL_RGBA8, GL_RGBA, GL_UNSIGNED_BYTE);

                // TODO: check completness and throw if not?
            } catch (glowl::FramebufferObjectException const& exc) {
                megamol::core::utility::log::Log::DefaultLog.WriteError(
                    "[ScreenShooter] Unable to create framebuffer object: %s\n", exc.what());
            }

            int xSteps = (data.imgWidth / data.tileWidth) + (((data.imgWidth % data.tileWidth) != 0) ? 1 : 0);
            int ySteps = (data.imgHeight / data.tileHeight) + (((data.imgHeight % data.tileHeight) != 0) ? 1 : 0);
            if (xSteps * ySteps == 0) {
                throw vislib::Exception("No tiles scheduled for rendering", __FILE__, __LINE__);
            }
            Log::DefaultLog.WriteMsg(Log::LEVEL_INFO + 100, "%d tile(s) scheduled for rendering", xSteps * ySteps);

            int vp[4];
            glGetIntegerv(GL_VIEWPORT, vp);

            // overlay for user information
            glBindFramebuffer(GL_FRAMEBUFFER, 0); // better safe then sorry, "unbind" fbo before delting one
            try {
                overlayfbo = std::make_shared<glowl::FramebufferObject>(
                    vp[0] + vp[2], vp[1] + vp[3], glowl::FramebufferObject::DEPTH24);
                overlayfbo->createColorAttachment(GL_RGBA8, GL_RGBA, GL_UNSIGNED_BYTE);

                // TODO: check completness and throw if not?
            } catch (glowl::FramebufferObjectException const& exc) {
                megamol::core::utility::log::Log::DefaultLog.WriteError(
                    "[ScreenShooter] Unable to create framebuffer object: %s\n", exc.what());
                overlayfbo = nullptr;
            }
            if (overlayfbo != nullptr) {
                crv.SetFramebuffer(overlayfbo);
                crv.SetTime(frameTime);
                view->OnRenderView(crv); // glClear by SFX
                /// view->Resize(static_cast<unsigned int>(vp[2]), static_cast<unsigned int>(vp[3]));
                glFlush();
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

                        tx = static_cast<float>(tileX) * static_cast<float>(overlayfbo->getWidth()) /
                             static_cast<float>(data.imgWidth);
                        ty = static_cast<float>(tileY) * static_cast<float>(overlayfbo->getHeight()) /
                             static_cast<float>(data.imgHeight);
                        tw = static_cast<float>(tileW) * static_cast<float>(overlayfbo->getWidth()) /
                             static_cast<float>(data.imgWidth);
                        th = static_cast<float>(tileH) * static_cast<float>(overlayfbo->getHeight()) /
                             static_cast<float>(data.imgHeight);

                        glDrawBuffer(GL_FRONT);
                        glMatrixMode(GL_PROJECTION);
                        glLoadIdentity();
                        glTranslatef(-1.0f, -1.0f, 0.0f);
                        glScalef(2.0f / static_cast<float>(overlayfbo->getWidth()),
                            2.0f / static_cast<float>(overlayfbo->getHeight()), 1.0f);
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
                    switch (backgroundMode) {
                    case 0:
                        crv.SetBackgroundColor(view->BackgroundColor());
                        break;
                    case 1:
                        crv.SetBackgroundColor(glm::vec4(0, 0, 0, 1));
                        break;
                    case 2:
                        crv.SetBackgroundColor(glm::vec4(1, 1, 1, 1));
                        break;
                    case 3:
                        crv.SetBackgroundColor(glm::vec4(0, 0, 0, 1));
                        break;
                    case 4:
                        if ((xi + yi) % 2) {
                            crv.SetBackgroundColor(glm::vec4(255.0f / 192.0f, 255.0f / 192.0f, 255.0f / 192.0f, 1));
                        } else {
                            crv.SetBackgroundColor(glm::vec4(255.0f / 128.0f, 255.0f / 128.0f, 255.0f / 128.0f, 1));
                        }
                        break;
                    default: /* don't set background */
                        break;
                    }

                    crv.SetFramebuffer(currentFbo);
                    crv.SetTime(frameTime);
                    // TODO how should the screenshot tiling be done in the furture?
                    //crv.SetTile(static_cast<float>(data.imgWidth), static_cast<float>(data.imgHeight),
                    //    static_cast<float>(tileX), static_cast<float>(tileY), static_cast<float>(tileW),
                    //    static_cast<float>(tileH));
                    view->OnRenderView(crv); // glClear by SFX
                    /// view->Resize(static_cast<unsigned int>(vp[2]), static_cast<unsigned int>(vp[3]));
                    glFlush();

                    currentFbo->bindToRead(0);
                    glGetError();
                    glReadPixels(0, 0, currentFbo->getWidth(), currentFbo->getHeight(),
                        (backgroundMode == 1) ? GL_RGBA : GL_RGB, GL_UNSIGNED_BYTE, buffer);
                    glBindFramebuffer(GL_READ_FRAMEBUFFER, 0);
                    if (glGetError() != GL_NO_ERROR) {
                        megamol::core::utility::log::Log::DefaultLog.WriteError(
                            "[ScreenShooter] Failed to create Screenshot: Cannot read image data.\n");
                    }
                    if (backgroundMode == 1) {
                        // fixing alpha from premultiplied to postmultiplied
                        for (UINT x = 0; x < data.tileWidth; x++) {
                            for (UINT y = 0; y < data.tileHeight; y++) {
                                BYTE* cptr = buffer + 4 * (x + y * data.tileWidth);
                                if (cptr[3] == 0)
                                    continue;
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

                        tx = static_cast<float>(tileX) * static_cast<float>(overlayfbo->getWidth()) /
                             static_cast<float>(data.imgWidth);
                        ty = static_cast<float>(tileY) * static_cast<float>(overlayfbo->getHeight()) /
                             static_cast<float>(data.imgHeight);
                        tw = static_cast<float>(tileW) * static_cast<float>(overlayfbo->getWidth()) /
                             static_cast<float>(data.imgWidth);
                        th = static_cast<float>(tileH) * static_cast<float>(overlayfbo->getHeight()) /
                             static_cast<float>(data.imgHeight);
                        tx -= 1.0f;
                        ty -= 1.0f;
                        tw += 2.0f;
                        th += 2.0f;

                        glDrawBuffer(GL_FRONT);
                        glMatrixMode(GL_PROJECTION);
                        glLoadIdentity();
                        glTranslatef(-1.0f, -1.0f, 0.0f);
                        glScalef(2.0f / static_cast<float>(overlayfbo->getWidth()),
                            2.0f / static_cast<float>(overlayfbo->getHeight()), 1.0f);
                        glMatrixMode(GL_MODELVIEW);
                        glLoadIdentity();
                        glDisable(GL_LIGHTING);
                        glDisable(GL_DEPTH_TEST);
                        glDisable(GL_BLEND);
                        glEnable(GL_TEXTURE_2D);
                        glDisable(GL_CULL_FACE);
                        overlayfbo->bindColorbuffer(0);

                        glBegin(GL_QUADS);
                        glTexCoord2f(tx / static_cast<float>(overlayfbo->getWidth()),
                            ty / static_cast<float>(overlayfbo->getHeight()));
                        glVertex2f(tx, ty);
                        glTexCoord2f(tx / static_cast<float>(overlayfbo->getWidth()),
                            (ty + th) / static_cast<float>(overlayfbo->getHeight()));
                        glVertex2f(tx, ty + th);
                        glTexCoord2f((tx + tw) / static_cast<float>(overlayfbo->getWidth()),
                            (ty + th) / static_cast<float>(overlayfbo->getHeight()));
                        glVertex2f(tx + tw, ty + th);
                        glTexCoord2f((tx + tw) / static_cast<float>(overlayfbo->getWidth()),
                            ty / static_cast<float>(overlayfbo->getHeight()));
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

            view->Resize(static_cast<unsigned int>(vp[2]), static_cast<unsigned int>(vp[3]));

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
    if (overlayfbo != nullptr) {
        overlayfbo.reset();
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
    } catch (...) {}
    try {
        file.Close();
    } catch (...) {}
    if (rollback) {
        try {
            if (vislib::sys::File::Exists(filename)) {
                vislib::sys::File::Delete(filename);
            }
        } catch (...) {}
    }
    if (data.tmpFiles[0] != NULL) {
        try {
            data.tmpFiles[0]->Close();
        } catch (...) {}
        delete data.tmpFiles[0];
    }
    if (data.tmpFiles[1] != NULL) {
        try {
            data.tmpFiles[1]->Close();
        } catch (...) {}
        delete data.tmpFiles[1];
    }
    delete[] buffer;
    currentFbo.reset();

    megamol::core::utility::log::Log::DefaultLog.WriteInfo("Screen shot stored");

    if (this->makeAnimSlot.Param<core::param::BoolParam>()->Value()) {
        if (this->animLastFrameTime >= this->animToSlot.Param<core::param::IntParam>()->Value()) {
            Log::DefaultLog.WriteInfo("Animation screen shots complete");

            // stop animation
            // core::param::ParamSlot *playSlot = dynamic_cast<core::param::ParamSlot*>(view->FindNamedObject("anim::play"));
            // if (playSlot != NULL)
            //    playSlot->Param<core::param::BoolParam>()->SetValue(false);
            this->outputCounter = 0;
        } else {
            core::param::ParamSlot* time = this->findTimeParam(view);
            if (time != NULL) {
                float nextTime = this->animLastFrameTime + this->animStepSlot.Param<core::param::FloatParam>()->Value();
                time->Param<core::param::FloatParam>()->SetValue(static_cast<float>(nextTime));
                closeAfter = false;

                view->RegisterHook(this); // ready for the next frame

            } else {
                this->makeAnimSlot.Param<core::param::BoolParam>()->SetValue(false);
                Log::DefaultLog.WriteInfo("Animation screen shooting aborted: unable to fetch time code");
            }
        }
    }

    if (closeAfter) {
        this->running = false;

        /// XXX TODO Tell any frontend service to shutdown
        Log::DefaultLog.WriteError("'close after' option is not yet supported for new megamol frontend.");
    }
}


/*
 * view::special::ScreenShooter::createScreenshot
 */
void view::special::ScreenShooter::createScreenshot(const std::string& filename) {
    this->imageFilenameSlot.Param<core::param::FilePathParam>()->SetValue(filename);

    triggerButtonClicked(this->triggerButtonSlot);
}


/*
 * view::special::ScreenShooter::triggerButtonClicked
 */
bool view::special::ScreenShooter::triggerButtonClicked(core::param::ParamSlot& slot) {
    // happy trigger finger hit button action happend
    using megamol::core::utility::log::Log;
    ASSERT(&slot == &this->triggerButtonSlot);

    vislib::StringA mvn(this->viewNameSlot.Param<core::param::StringParam>()->Value().c_str());
    Log::DefaultLog.WriteMsg(Log::LEVEL_INFO + 100, "ScreenShot of \"%s\" requested", mvn.PeekBuffer());

    vislib::sys::AutoLock lock(this->ModuleGraphLock());
    {
        core::ViewInstance* vi = nullptr;
        core::view::AbstractView* av = nullptr;

        auto& megamolgraph = frontend_resources.get<megamol::core::MegaMolGraph>();
        auto module_ptr = megamolgraph.FindModule(std::string(mvn.PeekBuffer()));
        vi = dynamic_cast<core::ViewInstance*>(module_ptr.get());
        av = dynamic_cast<core::view::AbstractView*>(module_ptr.get());

        if (vi != nullptr) {
            if (vi->View() != nullptr) {
                av = vi->View();
            }
        }
        if (av != nullptr) {
            if (this->makeAnimSlot.Param<core::param::BoolParam>()->Value()) {
                core::param::ParamSlot* timeSlot = this->findTimeParam(av);
                if (timeSlot != nullptr) {
                    auto startTime = static_cast<float>(this->animFromSlot.Param<core::param::IntParam>()->Value());
                    Log::DefaultLog.WriteInfo("Starting animation of screen shots at %f.", time);
                    timeSlot->Param<core::param::FloatParam>()->SetValue(startTime);
                    this->animLastFrameTime = std::numeric_limits<decltype(animLastFrameTime)>::lowest();
                } else {
                    Log::DefaultLog.WriteError("Unable to find animation time parameter in given view. Unable to make "
                                               "animation screen shots.");
                    this->makeAnimSlot.Param<core::param::BoolParam>()->SetValue(false);
                }
                // this is not a good idea because the animation module interferes with the "anim::time" parameter in
                // "play" mode ...
                // core::param::ParamSlot *playSlot =
                // dynamic_cast<core::param::ParamSlot*>(vi->View()->FindNamedObject("anim::play")); if (playSlot != NULL)
                //    playSlot->Param<core::param::BoolParam>()->SetValue(true);
            }
            av->RegisterHook(this);
        } else {
            // suppose a view was actually intended!
            bool found = false;
            const auto fun = [this, &found](core::view::AbstractView* v) {
                v->RegisterHook(this);
                found = true;
            };
            this->GetCoreInstance()->FindModuleNoLock<core::view::AbstractView>(mvn.PeekBuffer(), fun);
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
megamol::core::param::ParamSlot* view::special::ScreenShooter::findTimeParam(core::view::AbstractView* view) {
    vislib::TString name(this->animTimeParamNameSlot.Param<core::param::StringParam>()->Value().c_str());
    core::param::ParamSlot* timeSlot = nullptr;

    if (name.IsEmpty()) {
        timeSlot = dynamic_cast<core::param::ParamSlot*>(view->FindNamedObject("anim::time").get());
    } else {
        auto& megamolgraph = frontend_resources.get<megamol::core::MegaMolGraph>();
        std::string fullname = std::string(view->Name().PeekBuffer()) + "::" + std::string(name.PeekBuffer());
        timeSlot = megamolgraph.FindParameterSlot(fullname);
    }

    return timeSlot;
}
