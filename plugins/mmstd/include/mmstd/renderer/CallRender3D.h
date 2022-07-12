/**
 * MegaMol
 * Copyright (c) 2021, MegaMol Dev Team
 * All rights reserved.
 */

#pragma once

#include <glm/glm.hpp>

#include "mmcore/factories/CallAutoDescription.h"
#include "mmcore/view/CPUFramebuffer.h"
#include "mmstd/renderer/AbstractCallRender.h"

namespace megamol::core::view {

inline constexpr char callrender3d_name[] = "CallRender3D";

inline constexpr char callrender3d_desc[] = "CPU Rendering call";

using CallRender3D = BaseCallRender<CPUFramebuffer, callrender3d_name, callrender3d_desc>;

/** Description class typedef */
typedef factories::CallAutoDescription<CallRender3D> CallRender3DDescription;

} // namespace megamol::core::view
