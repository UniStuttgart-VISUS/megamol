/*
 * CallRender3DGL.h
 *
 * Copyright (C) 2018 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#pragma once

#include <glm/glm.hpp>
#include "mmcore/api/MegaMolCore.std.h"
#include "mmcore/factories/CallAutoDescription.h"
#include "mmcore/view/CallRenderGL.h"
#include "vislib/graphics/gl/FramebufferObject.h"

namespace megamol {
namespace core {
namespace view {
/**
 * New and improved base class of rendering graph calls
 *
 * Function "Render" tells the callee to render itself into the currently
 * active opengl context (TODO: Late on it could also be a FBO).
 *
 * Function "GetExtents" asks the callee to fill the extents member of the
 * call (bounding boxes, temporal extents).
 */
class CallRender3DGL : public CallRenderGL {
public:
    /**
     * Answer the name of the objects of this description.
     *
     * @return The name of the objects of this description.
     */
    static const char* ClassName(void) { return "CallRender3DGL"; }

    /**
     * Gets a human readable description of the module.
     *
     * @return A human readable description of the module.
     */
    static const char* Description(void) { return "New and improved call for rendering a frame"; }

    /**
     * Answer the number of functions used for this call.
     *
     * @return The number of functions used for this call.
     */
    static unsigned int FunctionCount(void) { return AbstractCallRender::FunctionCount(); }

    /**
     * Answer the name of the function used for this call.
     *
     * @param idx The index of the function to return it's name.
     *
     * @return The name of the requested function.
     */
    static const char* FunctionName(unsigned int idx) { return AbstractCallRender::FunctionName(idx); }

    /** Ctor. */
    CallRender3DGL(void) : CallRenderGL() { }

    /** Dtor. */
    virtual ~CallRender3DGL(void) = default;

    /**
     * Assignment operator
     *
     * @param rhs The right hand side operand
     *
     * @return A reference to this
     */
    CallRender3DGL& operator=(const CallRender3DGL& rhs) {
        CallRenderGL::operator=(rhs);
        return *this;
    }

};

/** Description class typedef */
typedef factories::CallAutoDescription<CallRender3DGL> CallRender3DGLDescription;

} // namespace view
} /* end namespace core */
} /* end namespace megamol */
