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
#include "glh/glh_genext.h"


#include "vislib/assert.h"
#include "vislib/File.h"
#include "vislib/memutils.h"
#include "vislib/Trace.h"


/*
 * AbstractOpenGLShader::CompileException::CompilationFailedAction
 */
vislib::graphics::gl::AbstractOpenGLShader::CompileException::CompileAction 
vislib::graphics::gl::AbstractOpenGLShader::CompileException
        ::CompilationFailedAction(GLenum type) {
    switch (type) {
        case GL_FRAGMENT_SHADER_ARB: return ACTION_COMPILE_FRAGMENT_CODE;
        case GL_GEOMETRY_SHADER_EXT: return ACTION_COMPILE_GEOMETRY_CODE;
        case GL_VERTEX_SHADER_ARB: return ACTION_COMPILE_VERTEX_CODE;
        case GL_COMPUTE_SHADER: return ACTION_COMPILE_COMPUTE_CODE;
        default: return ACTION_COMPILE_UNKNOWN;
    }
}


const char* vislib::graphics::gl::AbstractOpenGLShader::CompileException
        ::CompileActionName(CompileAction action) {
    switch (action) {
        case ACTION_UNKNOWN: return "unknown";
        case ACTION_COMPILE_UNKNOWN: return "compile unknown";
        case ACTION_COMPILE_VERTEX_CODE: return "compile vertex";
        case ACTION_COMPILE_FRAGMENT_CODE: return "compile fragment";
        case ACTION_COMPILE_GEOMETRY_CODE: return "compile geometry";
        case ACTION_COMPILE_COMPUTE_CODE: return "compile compute";
        case ACTION_LINK: return "link";
        default: return "unknown";
    }
}


/*
 * AbstractOpenGLShader::CompileException::CompileException
 */
vislib::graphics::gl::AbstractOpenGLShader::CompileException::CompileException(
        const char *file, const int line) : Exception(file, line), 
        action(ACTION_UNKNOWN) {
}


/*
 * AbstractOpenGLShader::CompileException::CompileException
 */
vislib::graphics::gl::AbstractOpenGLShader::CompileException::CompileException(
        const char *msg, const char *file, const int line) 
        : Exception(msg, file, line), action(ACTION_UNKNOWN) {
}


/*
 * AbstractOpenGLShader::CompileException::CompileException
 */
vislib::graphics::gl::AbstractOpenGLShader::CompileException::CompileException(
        const wchar_t *msg, const char *file, const int line) 
        : Exception(msg, file, line), action(ACTION_UNKNOWN) {
}


/*
 * AbstractOpenGLShader::CompileException::CompileException
 */
vislib::graphics::gl::AbstractOpenGLShader::CompileException::CompileException(
        const char *msg, CompileAction action, const char *file, 
        const int line) : Exception(msg, file, line), action(action) {
}


/*
 * AbstractOpenGLShader::CompileException::CompileException
 */
vislib::graphics::gl::AbstractOpenGLShader::CompileException::CompileException(
        const wchar_t *msg, CompileAction action, const char *file, 
        const int line) : Exception(msg, file, line), action(action) {
}


/*
 * AbstractOpenGLShader::CompileException::CompileException
 */
vislib::graphics::gl::AbstractOpenGLShader::CompileException::CompileException(
        const CompileException& rhs) : Exception(rhs), action(rhs.action) {
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
    this->action = rhs.action;
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
