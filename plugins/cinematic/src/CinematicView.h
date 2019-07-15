/*
 *CinematicView.h
*
* Copyright (C) 2018 by VISUS (Universitaet Stuttgart).
* Alle Rechte vorbehalten.
*/

#pragma once

#include "Cinematic/Cinematic.h"

#include "mmcore/CallerSlot.h"
#include "mmcore/view/View3D.h"

#include "mmcore/view/CallRender3D.h"
#include "mmcore/view/CallRenderView.h"

#include "mmcore/utility/SDFFont.h"

#include "mmcore/param/BoolParam.h"
#include "mmcore/param/ButtonParam.h"
#include "mmcore/param/EnumParam.h"
#include "mmcore/param/FilePathParam.h"
#include "mmcore/param/FloatParam.h"
#include "mmcore/param/IntParam.h"
#include "mmcore/view/CallRender3D.h"
#include "mmcore/view/Input.h"

#include "vislib/Serialisable.h"
#include "vislib/Trace.h"
#include "vislib/graphics/Camera.h"
#include "vislib/graphics/gl/FramebufferObject.h"
#include "vislib/graphics/gl/GLSLShader.h"
#include "vislib/math/Point.h"
#include "vislib/math/Rectangle.h"
#include "vislib/sys/FastFile.h"
#include "vislib/sys/Path.h"

#include "CallKeyframeKeeper.h"
#include "Keyframe.h"
#include "png.h"

namespace megamol {
namespace cinematic {

    /**
    * Cinemtic View.
    */
    class CinematicView : public core::view::View3D {
    public:

        typedef core::view::View3D Base;

        /**
         * Answer the name of this module.
         *
         * @return The name of this module.
         */
        static const char* ClassName(void) { return "CinematicView"; }

        /**
         * Answer a human readable description of this module.
         *
         * @return A human readable description of this module.
         */
        static const char* Description(void) { return "Screenshot View Module"; }

        /**
         * Answers whether this module is available on the current system.
         *
         * @return 'true' if the module is available, 'false' otherwise.
         */
        static bool IsAvailable(void) {
            if (!vislib::graphics::gl::FramebufferObject::AreExtensionsAvailable()) return false;
            return true;
        }

        /**
         * Disallow usage in quickstarts
         *
         * @return false
         */
        static bool SupportQuickstart(void) { return false; }

        /** Ctor. */
        CinematicView(void);

        /** Dtor. */
        virtual ~CinematicView(void);

    protected:

        /**
         * Renders this AbstractView3D in the currently active OpenGL context.
         */
        virtual void Render(const mmcRenderViewContext& context);

    private:

        typedef std::chrono::system_clock::time_point time_point;

        /**********************************************************************
         * variables
         **********************************************************************/

        megamol::core::utility::SDFFont theFont;

        enum SkyboxSides {
            SKYBOX_NONE = 0,
            SKYBOX_FRONT = 1,
            SKYBOX_BACK = 2,
            SKYBOX_LEFT = 4,
            SKYBOX_RIGHT = 8,
            SKYBOX_UP = 16,
            SKYBOX_DOWN = 32
        };

        clock_t deltaAnimTime;

        Keyframe shownKeyframe;
        bool playAnim;

        int cineWidth;
        int cineHeight;
        int vpHLast;
        int vpWLast;

        CinematicView::SkyboxSides sbSide;

        vislib::graphics::gl::FramebufferObject fbo;
        bool rendering;
        unsigned int fps;

        struct pngData {
            BYTE* buffer = nullptr;
            vislib::sys::FastFile file;
            unsigned int width;
            unsigned int height;
            unsigned int bpp;
            vislib::TString path;
            vislib::TString filename;
            unsigned int cnt;
            png_structp ptr = nullptr;
            png_infop infoptr = nullptr;
            float animTime;
            unsigned int write_lock;
            time_point start_time;
            unsigned int exp_frame_cnt;
        } pngdata;

        /**********************************************************************
         * functions
         **********************************************************************/

        /** 
        * Render to file functions 
        */
        bool render2file_setup();

        /**
        *
        */
        bool render2file_write_png();

        /**
        *
        */
        bool render2file_finish();

        /**
        *
        */
        bool setSimTime(float st);

        /**
         * My error handling function for png export
         *
         * @param pngPtr The png structure pointer
         * @param msg The error message
         */
        static void PNGAPI pngError(png_structp pngPtr, png_const_charp msg) {
            throw vislib::Exception(msg, __FILE__, __LINE__);
        }

        /**
         * My error handling function for png export
         *
         * @param pngPtr The png structure pointer
         * @param msg The error message
         */
        static void PNGAPI pngWarn(png_structp pngPtr, png_const_charp msg) {
            vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_WARN, "Png-Warning: %s\n", msg);
        }

        /**
         * My write function for png export
         *
         * @param pngPtr The png structure pointer
         * @param buf The pointer to the buffer to be written
         * @param size The number of bytes to be written
         */
        static void PNGAPI pngWrite(png_structp pngPtr, png_bytep buf, png_size_t size) {
            vislib::sys::File* f = static_cast<vislib::sys::File*>(png_get_io_ptr(pngPtr));
            f->Write(buf, size);
        }

        /**
         * My flush function for png export
         *
         * @param pngPtr The png structure pointer
         */
        static void PNGAPI pngFlush(png_structp pngPtr) {
            vislib::sys::File* f = static_cast<vislib::sys::File*>(png_get_io_ptr(pngPtr));
            f->Flush();
        }

        /**********************************************************************
         * callback
         **********************************************************************/

        /** The keyframe keeper caller slot */
        core::CallerSlot keyframeKeeperSlot;

        /**********************************************************************
         * parameters
         **********************************************************************/

        core::param::ParamSlot renderParam;
        core::param::ParamSlot delayFirstRenderFrameParam;
        core::param::ParamSlot startRenderFrameParam;
        core::param::ParamSlot toggleAnimPlayParam;
        core::param::ParamSlot selectedSkyboxSideParam;
        core::param::ParamSlot cubeModeRenderParam;
        core::param::ParamSlot resWidthParam;
        core::param::ParamSlot resHeightParam;
        core::param::ParamSlot fpsParam;
        core::param::ParamSlot eyeParam;
        core::param::ParamSlot projectionParam;
        core::param::ParamSlot frameFolderParam;
        core::param::ParamSlot addSBSideToNameParam;
    };

} /* end namespace cinematic */
} /* end namespace megamol */
