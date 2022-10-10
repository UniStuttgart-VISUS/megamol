/*
 * SimpleRenderTarget.h
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VISUS).
 * All rights reserved.
 */

#pragma once

#include <glowl/FramebufferObject.hpp>

#include "mmcore/CalleeSlot.h"
#include "mmstd/renderer/RendererModule.h"
#include "mmstd_gl/ModuleGL.h"
#include "mmstd_gl/renderer/CallRender3DGL.h"

namespace megamol {
namespace compositing_gl {

/**
 * TODO
 */
class SimpleRenderTarget : public core::view::RendererModule<mmstd_gl::CallRender3DGL, mmstd_gl::ModuleGL> {
public:
    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char* ClassName(void) {
        return "SimpleRenderTarget";
    }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char* Description(void) {
        return "Binds a FBO with color, normal and depth render targets.";
    }

    /**
     * Answers whether this module is available on the current system.
     *
     * @return 'true' if the module is available, 'false' otherwise.
     */
    static bool IsAvailable(void) {
        /*TODO*/
        return true;
    }

    /** Ctor. */
    SimpleRenderTarget();

    /** Dtor. */
    ~SimpleRenderTarget();

protected:
    /**
     * Implementation of 'Create'.
     *
     * @return 'true' on success, 'false' otherwise.
     */
    bool create();

    /**
     * Implementation of 'Release'.
     */
    void release();

    /**
     * The get extents callback. The module should set the members of
     * 'call' to tell the caller the extents of its data (bounding boxes
     * and times).
     *
     * @param call The calling call.
     *
     * @return The return value of the function.
     */
    bool GetExtents(mmstd_gl::CallRender3DGL& call);

    /**
     * The render callback.
     *
     * @param call The calling call.
     *
     * @return The return value of the function.
     */
    bool Render(mmstd_gl::CallRender3DGL& call);

    /**
     *
     */
    bool getColorRenderTarget(core::Call& caller);

    /**
     *
     */
    bool getNormalsRenderTarget(core::Call& caller);

    /**
     *
     */
    bool getDepthRenderTarget(core::Call& caller);

    /**
     *
     */
    bool getCameraSnapshot(core::Call& caller);

    /**
     *
     */
    bool getFramebufferObject(core::Call& caller);

    /**
     *
     */
    bool getMetaDataCallback(core::Call& caller);

    /**
     * G-Buffer for deferred rendering. By default if uses three color attachments (and a depth renderbuffer):
     * surface albedo - RGB 16bit per channel
     * normals - RGB 16bit per channel
     * depth - single channel 32bit
     */
    std::shared_ptr<glowl::FramebufferObject> m_GBuffer;

    uint32_t m_version;

private:
    /** Local copy of last used camera*/
    core::view::Camera m_last_used_camera;

    core::CalleeSlot m_color_render_target;
    core::CalleeSlot m_normal_render_target;
    core::CalleeSlot m_depth_render_target;

    /** Slot for accessing the camera that is propagated to the render chain from this module */
    core::CalleeSlot m_camera;

    /** Slot for accessing the framebuffer object used by this render target module */
    core::CalleeSlot m_framebuffer_slot;
};

} // namespace compositing_gl
} // namespace megamol
