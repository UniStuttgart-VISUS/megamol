/*
 * DeferredShading.h
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VISUS).
 * All rights reserved.
 */

#ifndef Deferred_SHADING_H_INCLUDED
#define Deferred_SHADING_H_INCLUDED

#include "vislib/math/Matrix.h"
#include "vislib_gl/graphics/gl/GLSLShader.h"

#include "mmcore/param/ParamSlot.h"
#include "mmcore_gl/view/CallRender3DGL.h"
#include "mmcore_gl/view/Renderer3DModuleGL.h"

#include "glowl/BufferObject.hpp"
#include "glowl/FramebufferObject.hpp"

namespace megamol {
namespace core_gl {

/**
 * TODO
 */
class DeferredShading : public megamol::core_gl::view::Renderer3DModuleGL {
public:
    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char* ClassName(void) {
        return "DeferredShading";
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
    DeferredShading();

    /** Dtor. */
    ~DeferredShading();

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
    bool GetExtents(core_gl::view::CallRender3DGL& call);

    /**
     * The render callback.
     *
     * @param call The calling call.
     *
     * @return The return value of the function.
     */
    bool Render(core_gl::view::CallRender3DGL& call);

    /**
     * Method that gets called before the rendering is started for all changed modules
     *
     * @param call The rendering call that contains the camera
     */
    void PreRender(core_gl::view::CallRender3DGL& call);

private:
    typedef vislib_gl::graphics::gl::GLSLShader GLSLShader;

    /** Shader program for deferred shading pass */
    std::unique_ptr<GLSLShader> m_deferred_shading_prgm;

    /**
     * G-Buffer for deferred rendering. By default if uses three color attachments (and a depth renderbuffer):
     * surface albedo - RGB 16bit per channel
     * normals - RGB 16bit per channel
     * depth - single channel 32bit
     */
    std::unique_ptr<glowl::FramebufferObject> m_GBuffer;

    /**
     * GPU buffer object for making active (point)lights available in during shading pass
     */
    std::unique_ptr<glowl::BufferObject> m_lights_buffer;

    /** The call for light sources */
    core::CallerSlot getLightsSlot;

    /** The btf file name */
    core::param::ParamSlot m_btf_filename_slot;
};

} // namespace core_gl
} // namespace megamol

#endif
