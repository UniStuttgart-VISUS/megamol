/*
 * CallRenderView.h
 *
 * Copyright (C) 2009 by VISUS (Universitaet Stuttgart).
 * Alle Rechte vorbehalten.
 */

#pragma once

#include "mmcore/factories/CallAutoDescription.h"
#include "mmcore/view/CPUFramebuffer.h"
#include "mmcore/view/Input.h"
#include "mmstd/renderer/AbstractCallRenderView.h"


namespace megamol {
namespace core {
namespace view {

inline constexpr char callrenderview_name[] = "CallRenderView";

inline constexpr char callrenderview_desc[] = "Call for rendering visual elements into a single target";

using CallRenderView = AbstractCallRenderView<CPUFramebuffer, callrenderview_name, callrenderview_desc>;

/** Description class typedef */
typedef factories::CallAutoDescription<CallRenderView> CallRenderViewDescription;


} /* end namespace view */
} /* end namespace core */
} /* end namespace megamol */
