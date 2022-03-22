/*
 * CallRenderViewGL.h
 *
 * Copyright (C) 2009 by VISUS (Universitaet Stuttgart).
 * Alle Rechte vorbehalten.
 */

#pragma once

#include "mmcore/factories/CallAutoDescription.h"
#include "mmcore/view/AbstractCallRenderView.h"
#include "mmcore/view/Input.h"

#include "glowl/FramebufferObject.hpp"


namespace megamol {
namespace core_gl {
namespace view {

inline constexpr char callrenderviewgl_name[] = "CallRenderViewGL";

inline constexpr char callrenderviewgl_desc[] = "Call for rendering visual elements into a single target";

/**
 * Call for rendering visual elements (from separate sources) into a single target, i.e.,
 * FBO-based compositing and cluster display.
 */
class CallRenderViewGL : public core::view::AbstractCallRenderView<glowl::FramebufferObject, callrenderviewgl_name,
                             callrenderviewgl_desc> {
public:
    /** Ctor. */
    CallRenderViewGL(void)
            : AbstractCallRenderView<glowl::FramebufferObject, callrenderviewgl_name, callrenderviewgl_desc>() {
        this->caps.RequireOpenGL();
    }

    /** Dtor. */
    virtual ~CallRenderViewGL(void) = default;
};

/** Description class typedef */
typedef core::factories::CallAutoDescription<CallRenderViewGL> CallRenderViewGLDescription;


} /* end namespace view */
} // namespace core_gl
} /* end namespace megamol */
