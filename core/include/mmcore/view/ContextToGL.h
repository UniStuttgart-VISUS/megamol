/*
 * ContextToGL.h
 *
 * Copyright (C) 2021 by VISUS (Universitaet Stuttgart).
 * Alle Rechte vorbehalten.
 */

#pragma once
#include "mmcore/view/Renderer3DModuleGL.h"
#include "vislib/graphics/gl/GLSLShader.h"
#include "mmcore/view/RenderUtils.h"
#include "mmcore/view/CallRender3D.h"

namespace megamol::core::view {

class ContextToGL : public Renderer3DModuleGL {
public:
    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char* ClassName(void) {
        return "ContextToGL";
    }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char* Description(void) {
        return "Merges content to the input GL buffer";
    }

    /**
     * Answers whether this module is available on the current system.
     *
     * @return 'true' if the module is available, 'false' otherwise.
     */
    static bool IsAvailable(void) {
        return vislib::graphics::gl::GLSLShader::AreExtensionsAvailable();
    }

    /** Ctor. */
    ContextToGL(void);

    /** Dtor. */
    virtual ~ContextToGL(void);

protected:

    /**
     * Implementation of 'Create'.
     *
     * @return 'true' on success, 'false' otherwise.
     */
    virtual bool create(void);

    /**
     * Implementation of 'Release'.
     */
    virtual void release(void);

    /**
     * The get extents callback. The module should set the members of
     * 'call' to tell the caller the extents of its data (bounding boxes
     * and times).
     *
     * @param call The calling call.
     *
     * @return The return value of the function.
     */
    virtual bool GetExtents(CallRender3DGL& call);

    /**
     * The render callback.
     *
     * @param call The calling call.
     *
     * @return The return value of the function.
     */
    virtual bool Render(CallRender3DGL& call);

private:

    core::CallerSlot _getContextSlot;

    std::shared_ptr<CPUFramebuffer> _framebuffer;

    RenderUtils _utils;

};



} // end namespace
