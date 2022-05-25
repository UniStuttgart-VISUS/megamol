/**
 * MegaMol
 * Copyright (c) 2018, MegaMol Dev Team
 * All rights reserved.
 */

#pragma once

#include "mmstd_gl/renderer/Renderer2DModuleGL.h"

namespace megamol::infovis_gl {

class Renderer2D : public mmstd_gl::Renderer2DModuleGL {
public:
    Renderer2D() : mmstd_gl::Renderer2DModuleGL() {}

    ~Renderer2D() override = default;

protected:
    static void computeDispatchSizes(
        uint64_t numItems, GLint const localSizes[3], GLint const maxCounts[3], GLuint dispatchCounts[3]);

    static inline void computeDispatchSizes(uint64_t numItems, std::array<GLint, 3> const& localSizes,
        std::array<GLint, 3> const& maxCounts, std::array<GLuint, 3>& dispatchCounts) {
        computeDispatchSizes(numItems, localSizes.data(), maxCounts.data(), dispatchCounts.data());
    }

    static std::tuple<double, double> mouseCoordsToWorld(
        double mouse_x, double mouse_y, core::view::Camera const& cam, int width, int height);

    static void makeDebugLabel(GLenum identifier, GLuint name, const char* label);
    static void debugNotify(GLuint name, const char* message);
    static void debugPush(GLuint name, const char* groupLabel);
    static void debugPop();
};

} // namespace megamol::infovis_gl
