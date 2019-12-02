/*
 * RenderMDIMesh.h
 *
 * Copyright (C) 2017 by Universitaet Stuttgart (VISUS).
 * All rights reserved.
 */

#ifndef NG_MESH_RENDERER_H_INCLUDED
#define NG_MESH_RENDERER_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#    pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "vislib/graphics/gl/GLSLShader.h"
#include "vislib/math/Matrix.h"

#include "mmcore/CallerSlot.h"
#include "mmcore/view/CallRender3D_2.h"
#include "mmcore/view/Renderer3DModule_2.h"

#include "compositing/CompositingCalls.h"

#include "mesh/GPUMaterialCollection.h"

#include "glowl/BufferObject.hpp"
#include "glowl/FramebufferObject.hpp"
#include "glowl/Mesh.hpp"

namespace megamol {
namespace mesh {


/**
 * Renderer module for rendering geometry with modern (OpenGL 4.3+) features.
 * Objects for rendering are supplied in batches. Each  render batch can contain
 * many objects that use the same shader program and also share the same geometry
 * or at least the same vertex format.
 * Per render batch, a single call of glMultiDrawElementsIndirect is made. The data
 * for the indirect draw call is stored and accessed via SSBOs.
 */
class RenderMDIMesh : public megamol::core::view::Renderer3DModule_2 {
public:
    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char* ClassName(void) { return "RenderMDIMesh"; }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char* Description(void) {
        return "Modern renderer for meshes. Objects are rendered in batches using indirect draw calls.";
    }

    /**
     * Answers whether this module is available on the current system.
     *
     * @return 'true' if the module is available, 'false' otherwise.
     */
    static bool IsAvailable(void) {
#ifdef _WIN32
#    if defined(DEBUG) || defined(_DEBUG)
        HDC dc = ::wglGetCurrentDC();
        HGLRC rc = ::wglGetCurrentContext();
        ASSERT(dc != NULL);
        ASSERT(rc != NULL);
#    endif // DEBUG || _DEBUG
#endif     // _WIN32
        return vislib::graphics::gl::GLSLShader::AreExtensionsAvailable() &&
               isExtAvailable("GL_ARB_shader_draw_parameters") && ogl_IsVersionGEQ(4, 3);
    }

    /** Ctor. */
    RenderMDIMesh();

    /** Dtor. */
    ~RenderMDIMesh();

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

private:
    /** GPU buffer object that stores per frame data, i.e. camera parameters */
    std::unique_ptr<glowl::BufferObject> m_per_frame_data;

    /** Optional render target framebuffer object */
    std::shared_ptr<glowl::FramebufferObject> m_render_target;

    megamol::core::CallerSlot m_render_task_callerSlot;

    megamol::core::CallerSlot m_framebuffer_slot;
};

} // namespace mesh
} // namespace megamol

#endif