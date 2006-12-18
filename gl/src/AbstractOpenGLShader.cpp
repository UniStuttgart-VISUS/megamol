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


/*
 * vislib::graphics::gl::AbstractOpenGLShader::read
 */
bool vislib::graphics::gl::AbstractOpenGLShader::read(StringA& outStr,
        const char *filename) const {
    using namespace vislib::sys;

    File file;                      // File to read source from.
    File::FileSize size;            // Size of the file in bytes.
    char *src = NULL;               // Array to hold source.

    /* Open shader file. */
    if (!file.Open(filename, File::READ_ONLY, File::SHARE_READ, 
            File::OPEN_ONLY)) {
        TRACE(Trace::LEVEL_ERROR, "Shader file \"%s\" could not be opened.", 
            filename);
        return false;
    }

    /* Allocate memory for source. */
    size = file.GetSize();
    ASSERT(size < INT_MAX);
    src = outStr.AllocateBuffer(static_cast<StringA::Size>(size));
    
    /* Read source and ensure binary zero at end. */
    file.Read(src, size); 
    src[size] = 0;

    return true;
}
