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
namespace core {
namespace view {

    inline constexpr char callrenderviewgl_name[] = "CallRenderViewGL";

    inline constexpr char callrenderviewgl_desc[] = "Call for rendering visual elements into a single target";

    /**
     * Call for rendering visual elements (from separate sources) into a single target, i.e.,
	 * FBO-based compositing and cluster display.
     */
    using CallRenderViewGL = AbstractCallRenderView<glowl::FramebufferObject, callrenderviewgl_name, callrenderviewgl_desc>;

    /** Description class typedef */
    typedef factories::CallAutoDescription<CallRenderViewGL>
        CallRenderViewGLDescription;


} /* end namespace view */
} /* end namespace core */
} /* end namespace megamol */

