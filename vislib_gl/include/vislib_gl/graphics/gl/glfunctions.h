/*
 * glfunctions.h
 *
 * Copyright (C) 2007 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */

#pragma once
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */

#include "vislib/math/Cuboid.h"


namespace vislib_gl::graphics::gl {

/**
 * Draws a cuboid using GL_LINES and immediate mode
 *
 * @param box The cuboid to be drawn
 */
void DrawCuboidLines(const vislib::math::Cuboid<int>& box);

/**
 * Draws a cuboid using GL_LINES and immediate mode
 *
 * @param box The cuboid to be drawn
 */
void DrawCuboidLines(const vislib::math::Cuboid<float>& box);

/**
 * Draws a cuboid using GL_LINES and immediate mode
 *
 * @param box The cuboid to be drawn
 */
void DrawCuboidLines(const vislib::math::Cuboid<double>& box);

} // namespace vislib_gl::graphics::gl

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
