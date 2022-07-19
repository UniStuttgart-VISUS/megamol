/*
 * DeferredShading.h
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VISUS).
 * All rights reserved.
 */

#pragma once

#include <glowl/BufferObject.hpp>
#include <glowl/Texture2D.hpp>

#include "mmcore/CallerSlot.h"
#include "mmstd_gl/renderer/CallRender3DGL.h"
#include "mmstd_gl/renderer/Renderer3DModuleGL.h"
#include "vislib/math/Matrix.h"
#include "vislib_gl/graphics/gl/GLSLShader.h"

namespace megamol {
namespace compositing {

/**
 * TODO
 */
class DrawToScreen : public megamol::mmstd_gl::Renderer3DModuleGL {
public:
    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char* ClassName(void) {
        return "DrawToScreen";
    }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char* Description(void) {
        return "...TODO...";
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
    DrawToScreen();

    /** Dtor. */
    ~DrawToScreen();

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
     * Method that gets called before the rendering is started for all changed modules
     *
     * @param call The rendering call that contains the camera
     */
    void PreRender(mmstd_gl::CallRender3DGL& call);

private:
    typedef vislib_gl::graphics::gl::GLSLShader GLSLShader;

    /** Dummy color texture to use when no texture is connected */
    std::shared_ptr<glowl::Texture2D> m_dummy_color_tx;

    /** Dummy depth texture to use when no depth texture is connected */
    std::shared_ptr<glowl::Texture2D> m_dummy_depth_tx;

    /** Shader program for deferred shading pass */
    std::unique_ptr<GLSLShader> m_drawToScreen_prgm;

    /** */
    core::CallerSlot m_input_texture_call;

    core::CallerSlot m_input_depth_texture_call;

    core::CallerSlot m_input_flags_call;

    glm::ivec2 m_last_tex_size = glm::ivec2(0, 0);
};

} // namespace compositing
} // namespace megamol
