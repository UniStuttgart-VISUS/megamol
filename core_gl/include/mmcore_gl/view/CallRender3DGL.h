/*
 * CallRender3DGL.h
 *
 * Copyright (C) 2018 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#pragma once

#include "mmcore/factories/CallAutoDescription.h"
#include "mmcore/view/CallRender3D.h"
#include <glm/glm.hpp>

#include <glowl/FramebufferObject.hpp>

namespace megamol {
namespace core_gl {
namespace view {

inline constexpr char callrender3dgl_name[] = "CallRender3DGL";

inline constexpr char callrender3dgl_desc[] = "New and improved call for rendering a frame with OpenGL";

/**
 * New and improved base class of rendering graph calls
 *
 * Function "Render" tells the callee to render itself into the currently
 * active opengl context (TODO: Late on it could also be a FBO).
 *
 * Function "GetExtents" asks the callee to fill the extents member of the
 * call (bounding boxes, temporal extents).
 */
class CallRender3DGL
        : public core::view::BaseCallRender<glowl::FramebufferObject, callrender3dgl_name, callrender3dgl_desc> {
public:
    /** Ctor. */
    CallRender3DGL(void) : BaseCallRender<glowl::FramebufferObject, callrender3dgl_name, callrender3dgl_desc>() {
        this->caps.RequireOpenGL();
    }

    /** Dtor. */
    virtual ~CallRender3DGL(void) = default;
};

/** Description class typedef */
typedef core::factories::CallAutoDescription<CallRender3DGL> CallRender3DGLDescription;

} // namespace view
} // namespace core_gl
} /* end namespace megamol */
