/*
 * DeferredShading.h
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VISUS).
 * All rights reserved.
 */

#ifndef DRAW_TO_SCREEN_H_INCLUDED
#define DRAW_TO_SCREEN_H_INCLUDED

#include "vislib/graphics/gl/GLSLShader.h"
#include "vislib/math/Matrix.h"

#include "mmcore/CallerSlot.h"
#include "mmcore/view/CallRender3D_2.h"
#include "mmcore/view/Renderer3DModule_2.h"

#include "glowl/BufferObject.hpp"

namespace megamol {
namespace compositing {

/**
 * TODO
 */
class DrawToScreen : public megamol::core::view::Renderer3DModule_2 {
public:
    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char* ClassName(void) { return "DrawToScreen"; }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char* Description(void) { return "...TODO..."; }

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
    bool GetExtents(core::view::CallRender3D_2& call);

    /**
     * The render callback.
     *
     * @param call The calling call.
     *
     * @return The return value of the function.
     */
    bool Render(core::view::CallRender3D_2& call);

    /**
     * Method that gets called before the rendering is started for all changed modules
     *
     * @param call The rendering call that contains the camera
     */
    void PreRender(core::view::CallRender3D_2& call);

private:
    typedef vislib::graphics::gl::GLSLShader GLSLShader;

    /** Shader program for deferred shading pass */
    std::unique_ptr<GLSLShader> m_drawToScreen_prgm;

    /** */
    core::CallerSlot m_input_texture_call;
};

} // namespace compositing
} // namespace megamol

#endif