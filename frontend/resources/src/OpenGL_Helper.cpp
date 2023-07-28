/**
 * MegaMol
 * Copyright (c) 2022, MegaMol Dev Team
 * All rights reserved.
 */

#include "OpenGL_Helper.h"

#ifdef MEGAMOL_USE_OPENGL_DEBUGGROUPS
#include <glad/gl.h>
#endif

void megamol::frontend_resources::OpenGL_Helper::PushDebugGroup(uint32_t id, int32_t length, std::string message) {
#ifdef MEGAMOL_USE_OPENGL_DEBUGGROUPS
    glPushDebugGroup(GL_DEBUG_SOURCE_APPLICATION, id, length, message.c_str());
#endif
}

void megamol::frontend_resources::OpenGL_Helper::PopDebugGroup() {
#ifdef MEGAMOL_USE_OPENGL_DEBUGGROUPS
    glPopDebugGroup();
#endif
}
