/**
 * MegaMol
 * Copyright (c) 2009, MegaMol Dev Team
 * All rights reserved.
 */

#pragma once

#include <glowl/FramebufferObject.hpp>

#include "mmcore/factories/CallAutoDescription.h"
#include "mmcore/view/AbstractCallRenderView.h"
#include "mmcore/view/Input.h"

namespace megamol::core_gl::view {

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
    CallRenderViewGL()
            : AbstractCallRenderView<glowl::FramebufferObject, callrenderviewgl_name, callrenderviewgl_desc>() {
        this->caps.RequireOpenGL();
    }

    /** Dtor. */
    ~CallRenderViewGL() override = default;
};

/** Description class typedef */
typedef core::factories::CallAutoDescription<CallRenderViewGL> CallRenderViewGLDescription;

} // namespace megamol::core_gl::view
