/**
 * MegaMol
 * Copyright (c) 2018, MegaMol Dev Team
 * All rights reserved.
 */

#pragma once

#include "mmcore_gl/view/Renderer2DModuleGL.h"

namespace megamol::infovis_gl {

#define DEBUG_NAME(name) name, (#name "\0")

class Renderer2D : public core_gl::view::Renderer2DModuleGL {
public:
    Renderer2D() : core_gl::view::Renderer2DModuleGL() {}

    ~Renderer2D() override = default;

protected:
    static void computeDispatchSizes(
        uint64_t numItems, GLint const localSizes[3], GLint const maxCounts[3], GLuint dispatchCounts[3]);

    static std::tuple<double, double> mouseCoordsToWorld(
        double mouse_x, double mouse_y, core::view::Camera const& cam, int width, int height);

    static void makeDebugLabel(GLenum identifier, GLuint name, const char* label);
    static void debugNotify(GLuint name, const char* message);
    static void debugPush(GLuint name, const char* groupLabel);
    static void debugPop();
};

} // namespace megamol::infovis_gl
