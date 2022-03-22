/*
 * ARBShader.cpp
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */

#ifdef _WIN32
#include <Windows.h>
#endif /* _WIN32 */

#include "vislib_gl/graphics/gl/ARBShader.h"

#include "vislib/IllegalParamException.h"
#include "vislib/IllegalStateException.h"
#include "vislib/UnsupportedOperationException.h"
#include "vislib/sys/sysfunctions.h"
#include "vislib_gl/graphics/gl/glverify.h"


/*
 * vislib_gl::graphics::gl::ARBShader::RequiredExtensions
 */
const char* vislib_gl::graphics::gl::ARBShader::RequiredExtensions(void) {
    return "GL_ARB_fragment_program "
           "GL_ARB_vertex_program ";
}


/*
 * vislib_gl::graphics::gl::ARBShader::ARBShader
 */
vislib_gl::graphics::gl::ARBShader::ARBShader(void) : AbstractOpenGLShader(), id(0), type(TYPE_UNKNOWN) {}


/*
 * vislib_gl::graphics::ARBShader::~ARBShader
 */
vislib_gl::graphics::gl::ARBShader::~ARBShader(void) {
    this->Release();
}


/*
 * vislib_gl::graphics::gl::ARBShader::Create
 */
bool vislib_gl::graphics::gl::ARBShader::Create(const char* src) {
    USES_GL_VERIFY;
    ASSERT(src != NULL);

    int errorPos = -1;

    /* Release possible old shader. */
    this->Release();

    /* Detect type of shader. */
    if (::strstr(src, ARBShader::FRAGMENT_SHADER_TOKEN) != NULL) {
        this->type = TYPE_FRAGMENT_SHADER;

    } else if (::strstr(src, ARBShader::VERTEX_SHADER_TOKEN) != NULL) {
        this->type = TYPE_VERTEX_SHADER;

    } else {
        this->type = TYPE_UNKNOWN;
        return false;
    }

    GL_VERIFY_THROW(::glEnable(this->type));
    GL_VERIFY_THROW(::glGenProgramsARB(1, &this->id));
    GL_VERIFY_THROW(::glBindProgramARB(this->type, this->id));
    GL_VERIFY_THROW(::glProgramStringARB(this->type, GL_PROGRAM_FORMAT_ASCII_ARB, static_cast<GLsizei>(::strlen(src)),
        reinterpret_cast<const GLubyte*>(src)));

    /* Get errors, if requested. */
    ::glGetIntegerv(GL_PROGRAM_ERROR_POSITION_ARB, &errorPos);
    if (errorPos != -1) {
        throw CompileException(
            reinterpret_cast<const char*>(::glGetString(GL_PROGRAM_ERROR_STRING_ARB)), __FILE__, __LINE__);
    }

    GL_VERIFY_THROW(::glDisable(this->type));

    return true;
}


/*
 * vislib_gl::graphics::gl::ARBShader::CreateFromFile
 */
bool vislib_gl::graphics::gl::ARBShader::CreateFromFile(const char* filename) {
    vislib::StringA src;

    if (vislib::sys::ReadTextFile(src, filename)) {
        return this->Create(src);
    } else {
        return false;
    }
}


/*
 * vislib_gl::graphics::gl::ARBShader::Disable
 */
GLenum vislib_gl::graphics::gl::ARBShader::Disable(void) {
    USES_GL_VERIFY;

    if (this->type != TYPE_UNKNOWN) {
        GL_VERIFY_RETURN(::glBindProgramARB(this->type, 0));
        GL_VERIFY_RETURN(::glDisable(this->type));
    }

    return GL_NO_ERROR;
}


/*
 * vislib_gl::graphics::gl::ARBShader::Enable
 */
GLenum vislib_gl::graphics::gl::ARBShader::Enable(void) {
    USES_GL_VERIFY;

    if (this->type != TYPE_UNKNOWN) {
        GL_VERIFY_RETURN(::glEnable(this->type));
        GL_VERIFY_RETURN(::glBindProgramARB(this->type, this->id));
        return GL_NO_ERROR;
    } else {
        throw vislib::IllegalStateException("'type' must not be TYPE_UNKNOWN", __FILE__, __LINE__);
    }
}


/*
 * vislib_gl::graphics::gl::ARBShader::Release
 */
GLenum vislib_gl::graphics::gl::ARBShader::Release(void) {
    USES_GL_VERIFY;

    this->type = TYPE_UNKNOWN;
    GL_VERIFY_RETURN(::glDeleteProgramsARB(1, &this->id));

    return GL_NO_ERROR;
}


/*
 * vislib_gl::graphics::gl::ARBShader::SetParameter
 */
GLenum vislib_gl::graphics::gl::ARBShader::SetParameter(
    const GLuint name, const double v1, const double v2, const double v3, const double v4) {
    USES_GL_VERIFY;

    GL_VERIFY_RETURN(::glProgramLocalParameter4dARB(this->type, name, v1, v2, v3, v4));
    return GL_NO_ERROR;
}


/*
 * vislib_gl::graphics::gl::ARBShader::SetParameter
 */
GLenum vislib_gl::graphics::gl::ARBShader::SetParameter(const GLuint name, const double* v) {
    USES_GL_VERIFY;
    ASSERT(v != NULL);

    GL_VERIFY_RETURN(::glProgramLocalParameter4dvARB(this->type, name, v));
    return GL_NO_ERROR;
}


/*
 * vislib_gl::graphics::gl::ARBShader::SetParameter
 */
GLenum vislib_gl::graphics::gl::ARBShader::SetParameter(
    const GLuint name, const float v1, const float v2, const float v3, const float v4) {
    USES_GL_VERIFY;

    GL_VERIFY_RETURN(::glProgramLocalParameter4fARB(this->type, name, v1, v2, v3, v4));
    return GL_NO_ERROR;
}


/*
 * vislib_gl::graphics::gl::ARBShader::SetParameter
 */
GLenum vislib_gl::graphics::gl::ARBShader::SetParameter(const GLuint name, const float* v) {
    USES_GL_VERIFY;
    ASSERT(v != NULL);

    GL_VERIFY_RETURN(::glProgramLocalParameter4fvARB(this->type, name, v));
    return GL_NO_ERROR;
}


/*
 * vislib_gl::graphics::gl::ARBShader::SetParameter
 */
GLenum vislib_gl::graphics::gl::ARBShader::SetParameter(const GLuint name, const int* v) {
    ASSERT(v != NULL);
    double vd[4];

    for (int i = 0; i < 4; i++) {
        vd[i] = static_cast<double>(v[i]);
    }

    return this->SetParameter(name, vd);
}


/*
 * vislib_gl::graphics::gl::ARBShader::FRAGMENT_SHADER_TOKEN
 */
const char* vislib_gl::graphics::gl::ARBShader::FRAGMENT_SHADER_TOKEN = "!!ARBfp";


/*
 * islib::graphics::gl::ARBShader::VERTEX_SHADER_TOKEN
 */
const char* vislib_gl::graphics::gl::ARBShader::VERTEX_SHADER_TOKEN = "!!ARBvp";


/*
 * vislib_gl::graphics::gl::ARBShader::ARBShader
 */
vislib_gl::graphics::gl::ARBShader::ARBShader(const ARBShader& rhs) {
    throw vislib::UnsupportedOperationException("ARBShader", __FILE__, __LINE__);
}


/*
 * vislib_gl::graphics::gl::ARBShader::operator =
 */
vislib_gl::graphics::gl::ARBShader& vislib_gl::graphics::gl::ARBShader::operator=(const ARBShader& rhs) {
    if (this != &rhs) {
        throw vislib::IllegalParamException("rhs", __FILE__, __LINE__);
    }

    return *this;
}
