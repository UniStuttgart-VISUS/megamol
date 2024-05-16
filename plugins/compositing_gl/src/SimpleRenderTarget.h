/*
 * SimpleRenderTarget.h
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VISUS).
 * All rights reserved.
 */

#pragma once

#include <glowl/FramebufferObject.hpp>

#include "CompositingOutHandler.h"
#include "mmcore/CalleeSlot.h"
#include "mmstd/renderer/RendererModule.h"
#include "mmstd_gl/ModuleGL.h"
#include "mmstd_gl/renderer/CallRender3DGL.h"

namespace megamol::compositing_gl {

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
    static const char* ClassName() {
        return "SimpleRenderTarget";
    }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char* Description() {
        return "Binds a FBO with color and normal render targets and a depth buffer.";
    }

    /**
     * Answers whether this module is available on the current system.
     *
     * @return 'true' if the module is available, 'false' otherwise.
     */
    static bool IsAvailable() {
        /*TODO*/
        return true;
    }

    /** Ctor. */
    SimpleRenderTarget();

    /** Dtor. */
    ~SimpleRenderTarget() override;

protected:
    /**
     * Implementation of 'Create'.
     *
     * @return 'true' on success, 'false' otherwise.
     */
    bool create() override;

    /**
     * Implementation of 'Release'.
     */
    void release() override;

    /**
     * The get extents callback. The module should set the members of
     * 'call' to tell the caller the extents of its data (bounding boxes
     * and times).
     *
     * @param call The calling call.
     *
     * @return The return value of the function.
     */
    bool GetExtents(mmstd_gl::CallRender3DGL& call) override;

    /**
     * The render callback.
     *
     * @param call The calling call.
     *
     * @return The return value of the function.
     */
    bool Render(mmstd_gl::CallRender3DGL& call) override;

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
    bool getDepthBuffer(core::Call& caller);

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
     * \brief Sets texture format variables.
     *
     *  @return 'true' if updates sucessfull, 'false' otherwise
     */
    bool textureFormatUpdate();

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
    core::CalleeSlot m_depth_buffer;

    /** Slot for accessing the camera that is propagated to the render chain from this module */
    core::CalleeSlot m_camera;

    /** Slot for accessing the framebuffer object used by this render target module */
    core::CalleeSlot m_framebuffer_slot;

    CompositingOutHandler colorOutHandler_;
    CompositingOutHandler normalsOutHandler_;
};

} // namespace megamol::compositing_gl
