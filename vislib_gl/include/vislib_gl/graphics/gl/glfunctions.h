/*
 * glfunctions.h
 *
 * Copyright (C) 2007 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */

#pragma once
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */

#include "vislib/VersionNumber.h"
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

/**
 * Enables or disables VSync. This method can only be used when there is a
 * current open gl context.
 *
 * @param enabled Flag whether to enable (a value of 'true') or disable (a
 *                value of 'false') VSync.
 *
 * @return true if the operation completed successfully, false otherwise.
 */
bool EnableVSync(bool enable = true);

/**
 * Disables VSync. This is identically to calling 'EnableVSync(false)'.
 * This method can only be used when there is a current open gl context.
 *
 * @return true if the operation completed successfully, false otherwise.
 */
inline bool DisableVSync() {
    // Note: not ordered correctly because we would need a forward
    //       declaration
    return EnableVSync(false);
}

/**
 * Answer the open gl version number. Note that the version number uses
 * only the first two or the first three number elements (where the third
 * element 'buildNumber' is used as release number).
 *
 * @return The open gl version number.
 */
const vislib::VersionNumber& GLVersion();

/**
 * Answer whether VSync is enabled or not. This method can only be used
 * when there is a current open gl context. If an error is returned the
 * status of VSync could not be read. However, enabling or disabling
 * VSync might still be possible using 'EnableVSync'.
 *
 * @param error Pointer to an bool receiving whether the function resulted
 *              in an error.
 *
 * @return 'true' if VSync is enabled, 'false' if not.
 */
bool IsVSyncEnabled(bool* error = NULL);

} // namespace vislib_gl::graphics::gl

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
