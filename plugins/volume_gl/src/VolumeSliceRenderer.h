/*
 * VolumeSliceRenderer.h
 *
 * Copyright (C) 2012-2019 by Universitaet Stuttgart (VISUS).
 * All rights reserved.
 */

#pragma once

#include "mmcore/CallerSlot.h"
#include "mmstd_gl/renderer/CallRender3DGL.h"
#include "mmstd_gl/renderer/Renderer3DModuleGL.h"

#include "vislib_gl/graphics/gl/GLSLComputeShader.h"
#include "vislib_gl/graphics/gl/GLSLShader.h"

namespace megamol {
namespace volume_gl {

/**
 * Renders one slice of a volume (slow)
 */
class VolumeSliceRenderer : public mmstd_gl::Renderer3DModuleGL {
public:
    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char* ClassName(void) {
        return "VolumeSliceRenderer_2";
    }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char* Description(void) {
        return "Renders one slice of a volume";
    }

    /**
     * Answers whether this module is available on the current system.
     *
     * @return 'true' if the module is available, 'false' otherwise.
     */
    static bool IsAvailable(void) {
        return true;
    }

    VolumeSliceRenderer(void);
    virtual ~VolumeSliceRenderer(void);

protected:
    /**
     * Implementation of 'Create'.
     *
     * @return 'true' on success, 'false' otherwise.
     */
    virtual bool create(void) override;

    /**
     * Implementation of 'Release'.
     */
    virtual void release(void) override;

    /**
     * The get extents callback. The module should set the members of
     * 'call' to tell the caller the extents of its data (bounding boxes
     * and times).
     *
     * @param call The calling call.
     *
     * @return The return value of the function.
     */
    virtual bool GetExtents(mmstd_gl::CallRender3DGL& call) override;

    /**
     * The render callback.
     *
     * @param call The calling call.
     *
     * @return The return value of the function.
     */
    virtual bool Render(mmstd_gl::CallRender3DGL& call) override;

private:
    /** The call for data */
    core::CallerSlot getVolSlot;

    /** The call for Transfer function */
    core::CallerSlot getTFSlot;

    /** The call for clipping plane */
    core::CallerSlot getClipPlaneSlot;

    /** Shader */
    vislib_gl::graphics::gl::GLSLComputeShader compute_shader;
    vislib_gl::graphics::gl::GLSLShader render_shader;
};

} // namespace volume_gl
} /* end namespace megamol */
