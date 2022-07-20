/**
 * MegaMol
 * Copyright (c) 2009, MegaMol Dev Team
 * All rights reserved.
 */

#pragma once

#include <glowl/FramebufferObject.hpp>

#include "mmcore/factories/CallAutoDescription.h"
#include "mmstd/renderer/CallRender3D.h"

namespace megamol::mmstd_gl {

inline constexpr char callrender2dgl_name[] = "CallRender2DGL";

inline constexpr char callrender2dgl_desc[] = "New and improved call for rendering a frame with OpenGL";

/**
 * Call for rendering 2d images
 *
 * Function "Render" tells the callee to render itself into the currently
 * active opengl context (TODO: Later on it could also be a FBO).
 * The bounding box member will be set to the world space rectangle
 * containing the visible part.
 *
 * Function "GetExtents" asks the callee to fill the extents member of the
 * call (bounding boxes).
 * The renderer should not draw anything outside the bounding box
 */
class CallRender2DGL
        : public core::view::BaseCallRender<glowl::FramebufferObject, callrender2dgl_name, callrender2dgl_desc> {
public:
    /** Ctor. */
    CallRender2DGL() : BaseCallRender<glowl::FramebufferObject, callrender2dgl_name, callrender2dgl_desc>() {
        this->caps.RequireOpenGL();
    }

    /** Dtor. */
    ~CallRender2DGL() override = default;
};

/** Description class typedef */
typedef core::factories::CallAutoDescription<CallRender2DGL> CallRender2DGLDescription;

} // namespace megamol::mmstd_gl
