/*
 *CinematicView.h
 *
 * Copyright (C) 2018 by VISUS (Universitaet Stuttgart).
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOL_CINEMATIC_CINEMATICVIEW_H_INCLUDED
#define MEGAMOL_CINEMATIC_CINEMATICVIEW_H_INCLUDED
#pragma once


#include "mmcore/CallerSlot.h"
#include "mmcore/MegaMolGraph.h"
#include "mmstd_gl/renderer/CallRender3DGL.h"
#include "mmstd_gl/renderer/CallRenderViewGL.h"
#include "mmstd_gl/view/View3DGL.h"

#include "cinematic/Keyframe.h"
#include "cinematic_gl/CinematicUtils.h"

#include <glm/gtx/quaternion.hpp>
#include <png.h>

#include "glowl/FramebufferObject.hpp"

#include "vislib/sys/FastFile.h"


namespace megamol {
namespace cinematic_gl {

/**
 * Cinemtic View.
 */
class CinematicView : public mmstd_gl::view::View3DGL {
public:
    typedef mmstd_gl::view::View3DGL Base;

    static void requested_lifetime_resources(frontend_resources::ResourceRequest& req) {
        Base::requested_lifetime_resources(req);
        req.require<core::MegaMolGraph>();
    }

    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char* ClassName(void) {
        return "CinematicView";
    }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char* Description(void) {
        return "Screenshot View Module";
    }

    /** Ctor. */
    CinematicView(void);

    /** Dtor. */
    virtual ~CinematicView(void);

protected:
    /**
     * Renders this View3DGL in the currently active OpenGL context.
     */
    virtual ImageWrapper Render(double time, double instanceTime) override;

private:
    typedef std::chrono::system_clock::time_point TimePoint_t;

    /**********************************************************************
     * variables
     **********************************************************************/

    enum SkyboxSides {
        SKYBOX_NONE = 0,
        SKYBOX_FRONT = 1,
        SKYBOX_BACK = 2,
        SKYBOX_LEFT = 4,
        SKYBOX_RIGHT = 8,
        SKYBOX_UP = 16,
        SKYBOX_DOWN = 32
    };

    struct PngData {
        BYTE* buffer = nullptr;
        vislib::sys::FastFile file;
        unsigned int width;
        unsigned int height;
        unsigned int bpp;
        vislib::StringA path;
        vislib::StringA filename;
        unsigned int cnt;
        png_structp structptr = nullptr;
        png_infop infoptr = nullptr;
        float animTime;
        unsigned int write_lock;
        TimePoint_t start_time;
        unsigned int exp_frame_cnt;
    };

    PngData png_data;
    CinematicUtils utils;
    clock_t deltaAnimTime;
    cinematic::Keyframe shownKeyframe;
    bool playAnim;
    int cineWidth;
    int cineHeight;
    SkyboxSides sbSide;
    bool rendering;
    unsigned int fps;
    bool skyboxCubeMode;
    std::shared_ptr<glowl::FramebufferObject> cinematicFbo;

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
        megamol::core::utility::log::Log::DefaultLog.WriteWarn("Png-Warning: %s\n", msg);
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
    core::param::ParamSlot frameFolderParam;
    core::param::ParamSlot addSBSideToNameParam;
};

} // namespace cinematic_gl
} /* end namespace megamol */

#endif // MEGAMOL_CINEMATIC_CINEMATICVIEW_H_INCLUDED
