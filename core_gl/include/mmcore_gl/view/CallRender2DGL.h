/*
 * CallRender2DGL.h
 *
 * Copyright (C) 2009 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_CALLRENDER2D_H_INCLUDED
#define MEGAMOLCORE_CALLRENDER2D_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "mmcore/factories/CallAutoDescription.h"
#include "mmcore/view/CallRender3D.h"

#include <glowl/FramebufferObject.hpp>

namespace megamol {
namespace core_gl {
namespace view {


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
    CallRender2DGL(void) : BaseCallRender<glowl::FramebufferObject, callrender2dgl_name, callrender2dgl_desc>() {
        this->caps.RequireOpenGL();
    }

    /** Dtor. */
    virtual ~CallRender2DGL(void) = default;
};


/** Description class typedef */
typedef core::factories::CallAutoDescription<CallRender2DGL> CallRender2DGLDescription;


} /* end namespace view */
} // namespace core_gl
} /* end namespace megamol */

#endif /* MEGAMOLCORE_CALLRENDER2D_H_INCLUDED */
