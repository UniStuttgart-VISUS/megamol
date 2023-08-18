//
// ogl_error_check.h
//
// Copyright (C) 2013 by University of Stuttgart (VISUS).
// All rights reserved.
//
// $Id$
//

#pragma once

#include "mmcore/utility/log/Log.h"
#include <GL/glu.h>

#define OGL_ERROR_CHECK // Toggle OpenGL error checking
#define GLSafeCall(err) glSafeCall(err, __FILE__, __LINE__)
#define CheckForGLError() checkForGLError(__FILE__, __LINE__)

/**
 * Utility function, that retrieves the last opengl error and prints an
 * error message if it is not GL_NO_ERROR.
 *
 * @param file The file in which the failure took place
 * @param line The line at which the failure took place
 * @return 'True' if the last error is GL_NO_ERROR, 'false' otherwise
 */
inline bool checkForGLError(const char* file, const int line) {
#ifdef OGL_ERROR_CHECK
    GLenum err;
    err = glGetError();
    if (err != GL_NO_ERROR) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "%s(%d) glError: %s", file, line, reinterpret_cast<char const*>(gluErrorString(err)));
        return false;
    }
#endif
    return true;
}

/**
 * Exits and prints an error message if a called method does return an
 * OpenGL error.
 *
 * @param err  The OpenGL related error
 * @param file The file in which the failure took place
 * @param line The line at which the failure took place
 * @return 'True' if the last error is GL_NO_ERROR, 'false' otherwise
 */
inline bool glSafeCall(GLenum err, const char* file, const int line) {
#ifdef OGL_ERROR_CHECK
    if (err != GL_NO_ERROR) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "%s(%d) glError: %s", file, line, reinterpret_cast<char const*>(gluErrorString(err)));
        return true;
    }
#endif
    return false;
}
