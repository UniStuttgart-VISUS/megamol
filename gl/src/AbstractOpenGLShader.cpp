/*
 * AbstractOpenGLShader.cpp
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */

#ifdef _WIN32
#include <windows.h>
#endif /* _WIN32 */

#include "vislib/AbstractOpenGLShader.h"

#include <climits>

#include "vislib/assert.h"
#include "vislib/File.h"
#include "vislib/memutils.h"
#include "vislib/Trace.h"


/*
 * AbstractOpenGLShader::CompileException::CompileException
 */
vislib::graphics::gl::AbstractOpenGLShader::CompileException::CompileException(
        const char *file, const int line) : Exception(file, line) {
}


/*
 * AbstractOpenGLShader::CompileException::CompileException
 */
vislib::graphics::gl::AbstractOpenGLShader::CompileException::CompileException(
        const char *msg, const char *file, const int line) 
        : Exception(msg, file, line) {
}


/*
 * AbstractOpenGLShader::CompileException::CompileException
 */
vislib::graphics::gl::AbstractOpenGLShader::CompileException::CompileException(
        const wchar_t *msg, const char *file, const int line) 
        : Exception(msg, file, line) {
}


/*
 * AbstractOpenGLShader::CompileException::CompileException
 */
vislib::graphics::gl::AbstractOpenGLShader::CompileException::CompileException(
        const CompileException& rhs) : Exception(rhs) {
}


/*
 * AbstractOpenGLShader::CompileException::~CompileException
 */
vislib::graphics::gl::AbstractOpenGLShader::CompileException::~CompileException(
        void) {
}


/*
 * AbstractOpenGLShader::CompileException::operator =
 */
vislib::graphics::gl::AbstractOpenGLShader::CompileException& 
vislib::graphics::gl::AbstractOpenGLShader::CompileException::operator =(
        const CompileException& rhs) {
    Exception::operator =(rhs);
    return *this;
}


/*
 * vislib::graphics::gl::AbstractOpenGLShader::~AbstractOpenGLShader
 */
vislib::graphics::gl::AbstractOpenGLShader::~AbstractOpenGLShader(void) {
    /* Nothing to do. */
}


/*
 * vislib::graphics::gl::AbstractOpenGLShader::AbstractOpenGLShader
 */
vislib::graphics::gl::AbstractOpenGLShader::AbstractOpenGLShader(void) {
    /* Nothing to do. */
}
