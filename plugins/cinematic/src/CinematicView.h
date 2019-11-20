/*
 *CinematicView.h
*
* Copyright (C) 2018 by VISUS (Universitaet Stuttgart).
* Alle Rechte vorbehalten.
*/

#ifndef MEGAMOL_CINEMATIC_CINEMATICVIEW_H_INCLUDED
#define MEGAMOL_CINEMATIC_CINEMATICVIEW_H_INCLUDED

#include "Cinematic/Cinematic.h"

#include "mmcore/CallerSlot.h"
#include "mmcore/view/View3D_2.h"
#include "mmcore/view/CallRender3D_2.h"
#include "mmcore/view/CallRenderView.h"
#include "mmcore/utility/SDFFont.h"
#include "mmcore/param/BoolParam.h"
#include "mmcore/param/ButtonParam.h"
#include "mmcore/param/EnumParam.h"
#include "mmcore/param/FilePathParam.h"
#include "mmcore/param/FloatParam.h"
#include "mmcore/param/IntParam.h"
#include "mmcore/view/Input.h"

#include "vislib/Serialisable.h"
#include "vislib/Trace.h"
#include "vislib/graphics/gl/FramebufferObject.h"
#include "vislib/math/Point.h"
#include "vislib/math/Rectangle.h"
#include "vislib/sys/FastFile.h"
#include "vislib/sys/Path.h"

#include "png.h"
#include "Keyframe.h"
#include "CinematicUtils.h"
#include "CallKeyframeKeeper.h"

#include <glm/gtx/quaternion.hpp>


namespace megamol {
namespace cinematic {

    /**
    * Cinemtic View.
    */
    class CinematicView : public core::view::View3D_2 {
    public:

        typedef core::view::View3D_2 Base;

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
         * Renders this View3D_2 in the currently active OpenGL context.
         */
        virtual void Render(const mmcRenderViewContext& context);

    private:

        typedef std::chrono::system_clock::time_point time_point;

        /**********************************************************************
         * variables
         **********************************************************************/

        enum SkyboxSides {
            SKYBOX_NONE  = 0,
            SKYBOX_FRONT = 1,
            SKYBOX_BACK  = 2,
            SKYBOX_LEFT  = 4,
            SKYBOX_RIGHT = 8,
            SKYBOX_UP    = 16,
            SKYBOX_DOWN  = 32
        };

        struct PngData {
            BYTE*                 buffer = nullptr;
            vislib::sys::FastFile file;
            unsigned int          width;
            unsigned int          height;
            unsigned int          bpp;
            vislib::StringA       path;
            vislib::StringA       filename;
            unsigned int          cnt;
            png_structp           structptr = nullptr;
            png_infop             infoptr = nullptr;
            float                 animTime;
            unsigned int          write_lock;
            time_point            start_time;
            unsigned int          exp_frame_cnt;
        };

        vislib::graphics::gl::FramebufferObject fbo;
        PngData                                 png_data;
        CinematicUtils                          utils;
        clock_t                                 deltaAnimTime;
        Keyframe                                shownKeyframe;
        bool                                    playAnim;
        int                                     cineWidth;
        int                                     cineHeight;
        float                                   vp_lastw;
        float                                   vp_lasth;
        SkyboxSides                             sbSide;
        bool                                    rendering;
        unsigned int                            fps;
        bool                                    skyboxCubeMode;

        /**********************************************************************
         * functions
         **********************************************************************/

        // PNG ----------------------------------------------------------------

        bool render_to_file_setup();

        bool render_to_file_write();

        bool render_to_file_cleanup();

        /**
         * Error handling function for png export
         *
         * @param pngPtr The png structure pointer
         * @param msg The error message
         */
        static void PNGAPI pngError(png_structp pngPtr, png_const_charp msg) {
            throw vislib::Exception(msg, __FILE__, __LINE__);
        }

        /**
         * Warning handling function for png export
         *
         * @param pngPtr The png structure pointer
         * @param msg The error message
         */
        static void PNGAPI pngWarn(png_structp pngPtr, png_const_charp msg) {
            vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_WARN, "Png-Warning: %s\n", msg);
        }

        /**
         * Write function for png export
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
         * Flush function for png export
         *
         * @param pngPtr The png structure pointer
         */
        static void PNGAPI pngFlush(png_structp pngPtr) {
            vislib::sys::File* f = static_cast<vislib::sys::File*>(png_get_io_ptr(pngPtr));
            f->Flush();
        }

        /**********************************************************************
         * callbacks
         **********************************************************************/

        core::CallerSlot keyframeKeeperSlot;

        /**********************************************************************
         * parameters
         **********************************************************************/

        core::param::ParamSlot renderParam;
        core::param::ParamSlot delayFirstRenderFrameParam;
        core::param::ParamSlot firstRenderFrameParam;
        core::param::ParamSlot lastRenderFrameParam;
        core::param::ParamSlot toggleAnimPlayParam;
        core::param::ParamSlot selectedSkyboxSideParam;
        core::param::ParamSlot skyboxCubeModeParam;
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

#endif // MEGAMOL_CINEMATIC_CINEMATICVIEW_H_INCLUDED