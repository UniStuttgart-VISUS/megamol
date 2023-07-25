/**
 * MegaMol
 * Copyright (c) 2022, MegaMol Dev Team
 * All rights reserved.
 */

#pragma once

#include <string>

namespace megamol::frontend_resources {

static std::string OpenGL_Helper_Req_Name = "OpenGL_Helper";

struct OpenGL_Helper {
    void PushDebugGroup(uint32_t id, int32_t length, std::string message);
    void PopDebugGroup();
};

} // namespace megamol::frontend_resources
