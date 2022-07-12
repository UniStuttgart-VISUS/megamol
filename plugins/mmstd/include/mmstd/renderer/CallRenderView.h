/**
 * MegaMol
 * Copyright (c) 2009, MegaMol Dev Team
 * All rights reserved.
 */

#pragma once

#include "mmcore/factories/CallAutoDescription.h"
#include "mmcore/view/CPUFramebuffer.h"
#include "mmcore/view/Input.h"
#include "mmstd/renderer/AbstractCallRenderView.h"

namespace megamol::core::view {

inline constexpr char callrenderview_name[] = "CallRenderView";

inline constexpr char callrenderview_desc[] = "Call for rendering visual elements into a single target";

using CallRenderView = AbstractCallRenderView<CPUFramebuffer, callrenderview_name, callrenderview_desc>;

/** Description class typedef */
typedef factories::CallAutoDescription<CallRenderView> CallRenderViewDescription;

} // namespace megamol::core::view
