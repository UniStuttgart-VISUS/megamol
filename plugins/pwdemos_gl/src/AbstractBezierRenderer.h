/*
 * AbstractBezierRenderer.h
 *
 * Copyright (C) 2018 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#pragma once


#include "mmcore/CallerSlot.h"
#include "mmstd_gl/renderer/CallRender3DGL.h"
#include "mmstd_gl/renderer/Renderer3DModuleGL.h"
#include "vislib_gl/graphics/gl/GLSLShader.h"


namespace megamol {
namespace demos_gl {

/**
 * Raycasting-based renderer for bézier curve tubes
 */
class AbstractBezierRenderer : public mmstd_gl::Renderer3DModuleGL {
public:
protected:
    /** Ctor. */
    AbstractBezierRenderer(void);

    /** Dtor. */
    virtual ~AbstractBezierRenderer(void);

    /**
     * Implementation of 'Create'.
     *
     * @return 'true' on success, 'false' otherwise.
     */
    virtual bool create(void);

    /**
     * The get extents callback. The module should set the members of
     * 'call' to tell the caller the extents of its data (bounding boxes
     * and times).
     *
     * @param call The calling call.
     *
     * @return The return value of the function.
     */
    virtual bool GetExtents(mmstd_gl::CallRender3DGL& call);

    /**
     * Implementation of 'Release'.
     */
    virtual void release(void);

    /**
     * The render callback.
     *
     * @param call The calling call.
     *
     * @return The return value of the function.
     */
    virtual bool Render(mmstd_gl::CallRender3DGL& call);

    /**
     * The implementation of the render callback
     *
     * @param call The calling rendering call
     *
     * @return The return value of the function
     */
    virtual bool render(mmstd_gl::CallRender3DGL& call) = 0;

    /**
     * Informs the class if the shader is required
     *
     * @return True if the shader is required
     */
    virtual bool shader_required(void) const {
        // TODO: This is not cool at all
        return true;
    }

    /** The call for data */
    core::CallerSlot getDataSlot;

    /** The data hash of the objects stored in the list */
    SIZE_T objsHash;

    /** The selected shader */
    vislib_gl::graphics::gl::GLSLShader* shader;

private:
};

} // namespace demos_gl
} /* end namespace megamol */
