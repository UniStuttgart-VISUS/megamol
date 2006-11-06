/*
 * ARBShader.cpp
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */

#ifdef _WIN32
#include <windows.h>
#endif /* _WIN32 */

#include "vislib/ARBShader.h"

#include "vislib/glverify.h"
#include "vislib/IllegalParamException.h"
#include "vislib/IllegalStateException.h"
#include "vislib/UnsupportedOperationException.h"


/*
 * vislib::graphics::gl::ARBShader::InitialiseExtensions
 */
bool vislib::graphics::gl::ARBShader::InitialiseExtensions(void) {
    return (::glh_init_extensions(
        "GL_ARB_fragment_program "
        "GL_ARB_vertex_program "
        ) != 0);
}


/*
 * vislib::graphics::gl::ARBShader::ARBShader
 */
vislib::graphics::gl::ARBShader::ARBShader(void) 
        : AbstractOpenGLShader(), id (0), type(TYPE_UNKNOWN) {
}


/*
 * vislib::graphics::ARBShader::~ARBShader
 */
vislib::graphics::gl::ARBShader::~ARBShader(void) {
    this->Release();
}


/*
 * vislib::graphics::gl::ARBShader::Create
 */
bool vislib::graphics::gl::ARBShader::Create(const char *src) {
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
    GL_VERIFY_THROW(::glProgramStringARB(this->type, 
        GL_PROGRAM_FORMAT_ASCII_ARB, static_cast<GLsizei>(::strlen(src)),
        reinterpret_cast<const GLubyte *>(src)));

    /* Get errors, if requested. */
    ::glGetIntegerv(GL_PROGRAM_ERROR_POSITION_ARB, &errorPos);
    if (errorPos != -1) {
        throw CompileException(reinterpret_cast<const char *>(
            ::glGetString(GL_PROGRAM_ERROR_STRING_ARB)), __FILE__, __LINE__);
    }

    GL_VERIFY_THROW(::glDisable(this->type));

    return true;
}


/*
 * vislib::graphics::gl::ARBShader::CreateFromFile
 */
bool vislib::graphics::gl::ARBShader::CreateFromFile(const char *filename) {
    StringA src;

    if (this->read(src, filename)) {
        return this->Create(src);
    } else {
        return false;
    }
}


/*
 * vislib::graphics::gl::ARBShader::Disable
 */
void vislib::graphics::gl::ARBShader::Disable(void) {
    if (this->type != TYPE_UNKNOWN) {
        ::glBindProgramARB(this->type, 0);
        ::glDisable(this->type);
    }
}


/*
 * vislib::graphics::gl::ARBShader::Enable
 */
void vislib::graphics::gl::ARBShader::Enable(void) {
    USES_GL_VERIFY;

    if (this->type != TYPE_UNKNOWN) {
        ::glEnable(this->type);
        GL_VERIFY_THROW(::glBindProgramARB(this->type, this->id));
    } else {
        throw IllegalStateException("'type' must not be TYPE_UNKNOWN",
            __FILE__, __LINE__);
    }
}


/*
 * vislib::graphics::gl::ARBShader::Release
 */
void vislib::graphics::gl::ARBShader::Release(void) {
    ::glDeleteProgramsARB(1, &this->id);
    this->type = TYPE_UNKNOWN;
}


/*
 * vislib::graphics::gl::ARBShader::FRAGMENT_SHADER_TOKEN
 */
const char *vislib::graphics::gl::ARBShader::FRAGMENT_SHADER_TOKEN = "!!ARBfp";


/*
 * islib::graphics::gl::ARBShader::VERTEX_SHADER_TOKEN
 */
const char *vislib::graphics::gl::ARBShader::VERTEX_SHADER_TOKEN = "!!ARBvp";


/*
 * vislib::graphics::gl::ARBShader::ARBShader
 */
vislib::graphics::gl::ARBShader::ARBShader(const ARBShader& rhs) {
    throw UnsupportedOperationException("ARBShader", __FILE__, __LINE__);
}


/*
 * vislib::graphics::gl::ARBShader::operator =
 */
vislib::graphics::gl::ARBShader& vislib::graphics::gl::ARBShader::operator =(
        const ARBShader& rhs) {
    if (this != &rhs) {
        throw IllegalParamException("rhs", __FILE__, __LINE__);
    }

    return *this;
}