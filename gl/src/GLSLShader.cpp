/*
 * GLSLShader.cpp
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */

#ifdef _WIN32
#include <windows.h>
#endif /* _WIN32 */

#include "vislib/GLSLShader.h"

#include "vislib/glverify.h"
#include "vislib/IllegalParamException.h"
#include "vislib/IllegalStateException.h"
#include "vislib/UnsupportedOperationException.h"


/*
 * vislib::graphics::gl::GLSLShader::InitialiseExtensions
 */
bool vislib::graphics::gl::GLSLShader::InitialiseExtensions(void) {
    return (::glh_init_extensions(
        "GL_ARB_shader_objects "
        ) != 0);
}


/*
 * vislib::graphics::gl::GLSLShader::FTRANSFORM_VERTEX_SHADER_SRC
 */
const char *vislib::graphics::gl::GLSLShader::FTRANSFORM_VERTEX_SHADER_SRC =
    "void main() { "
    "    gl_Position = ftransform();"
    "}";


/*
 * vislib::graphics::gl::GLSLShader::GLSLShader
 */
vislib::graphics::gl::GLSLShader::GLSLShader(void) 
        : AbstractOpenGLShader(), hProgObj(0) {
}


/*
 * vislib::graphics::gl::GLSLShader::~GLSLShader
 */
vislib::graphics::gl::GLSLShader::~GLSLShader(void) {
    this->Release();
}


/*
 * vislib::graphics::gl::GLSLShader::Create
 */
bool vislib::graphics::gl::GLSLShader::Create(const char *vertexShaderSrc, 
                                              const char *fragmentShaderSrc) {
    const char *v[] = { vertexShaderSrc };
    const char *f[] = { fragmentShaderSrc };
    
    return this->Create(v, 1, f, 1);
}

/*
 * vislib::graphics::gl::GLSLShader::Create
 */
bool vislib::graphics::gl::GLSLShader::Create(const char **vertexShaderSrc, 
        const SIZE_T cntVertexShaderSrc, const char **fragmentShaderSrc,
        const SIZE_T cntFragmentShaderSrc) {
    USES_GL_VERIFY;
    ASSERT(vertexShaderSrc != NULL);
    ASSERT(fragmentShaderSrc != NULL);

    GLhandleARB hPixelShader;
    GLhandleARB hVertexShader;

    /* Prepare vertex shader. */
    GL_VERIFY_THROW(hVertexShader = ::glCreateShaderObjectARB(
        GL_VERTEX_SHADER_ARB));
    GL_VERIFY_THROW(::glShaderSourceARB(hVertexShader, 
        static_cast<GLsizei>(cntVertexShaderSrc), vertexShaderSrc, NULL));
    GL_VERIFY_THROW(::glCompileShaderARB(hVertexShader));
    if (!this->isCompiled(hVertexShader)) {
        throw CompileException(this->getProgramInfoLog(hVertexShader), __FILE__,
            __LINE__);
    }

    /* Prepare pixel shader. */
    GL_VERIFY_THROW(hPixelShader = ::glCreateShaderObjectARB(
        GL_FRAGMENT_SHADER_ARB));
    GL_VERIFY_THROW(::glShaderSourceARB(hPixelShader, 
        static_cast<GLsizei>(cntFragmentShaderSrc), fragmentShaderSrc, NULL));
    GL_VERIFY_THROW(::glCompileShaderARB(hPixelShader));
    if (!this->isCompiled(hVertexShader)) {
        throw CompileException(this->getProgramInfoLog(hPixelShader), __FILE__, 
            __LINE__);
    }

    /* Assemble program object. */
    GL_VERIFY_THROW(this->hProgObj = ::glCreateProgramObjectARB());
    GL_VERIFY_THROW(::glAttachObjectARB(this->hProgObj, hVertexShader));
    GL_VERIFY_THROW(::glAttachObjectARB(this->hProgObj, hPixelShader));
    GL_VERIFY_THROW(::glLinkProgramARB(this->hProgObj));
    if (!this->isLinked(this->hProgObj)) {
        throw CompileException(this->getProgramInfoLog(this->hProgObj), 
            __FILE__, __LINE__);
    }

    return true;
}


/*
 * vislib::graphics::gl::GLSLShader::CreateFromFile
 */
bool vislib::graphics::gl::GLSLShader::CreateFromFile(
        const char *vertexShaderFile, const char *fragmentShaderFile) {
    StringA vertexShaderSrc;
    StringA fragmentShaderSrc;

    if (!this->read(vertexShaderSrc, vertexShaderFile)) {
        return false;
    }

    if (!this->read(fragmentShaderSrc, fragmentShaderFile)) {
        return false;
    }

    return this->Create(vertexShaderSrc, fragmentShaderSrc);
}


/*
 * vislib::graphics::gl::GLSLShader::Disable
 */
void vislib::graphics::gl::GLSLShader::Disable(void) {
    ::glUseProgramObjectARB(0);
    ::glDisable(GL_VERTEX_PROGRAM_ARB);
    ::glDisable(GL_FRAGMENT_PROGRAM_ARB);
}
        

/*
 * vislib::graphics::gl::GLSLShader::Enable
 */
void vislib::graphics::gl::GLSLShader::Enable(void) {
    USES_GL_VERIFY;
    GL_VERIFY_THROW(::glEnable(GL_VERTEX_PROGRAM_ARB));
    GL_VERIFY_THROW(::glEnable(GL_FRAGMENT_PROGRAM_ARB));
    GL_VERIFY_THROW(::glUseProgramObjectARB(this->hProgObj));
}


/*
 * vislib::graphics::gl::GLSLShader::Release
 */
void vislib::graphics::gl::GLSLShader::Release(void) {
    ::glDeleteObjectARB(this->hProgObj);
}


/*
 * vislib::graphics::gl::GLSLShader::getProgramInfoLog
 */
vislib::StringA vislib::graphics::gl::GLSLShader::getProgramInfoLog(
        GLhandleARB hProg) {
    USES_GL_VERIFY;
    GLint len = 0;
    GLint written = 0;
    StringA retval;
    char *log = NULL;

    GL_VERIFY_THROW(::glGetObjectParameterivARB(hProg, 
        GL_OBJECT_INFO_LOG_LENGTH_ARB, &len));

    if (len > 0) {
        log = retval.AllocateBuffer(len);
        GL_VERIFY_THROW(::glGetInfoLogARB(hProg, len, &written, log));
    }

    return retval;
}


/*
 * vislib::graphics::gl::GLSLShader::isCompiled
 */
bool vislib::graphics::gl::GLSLShader::isCompiled(GLhandleARB hProg) {
    USES_GL_VERIFY;
    GLint status;

    GL_VERIFY_THROW(::glGetObjectParameterivARB(hProg, 
        GL_OBJECT_COMPILE_STATUS_ARB, &status));

    return (status != GL_FALSE);
}


/*
 * vislib::graphics::gl::GLSLShader::isLinked
 */
bool vislib::graphics::gl::GLSLShader::isLinked(GLhandleARB hProg) {
    USES_GL_VERIFY;
    GLint status;

    GL_VERIFY_THROW(::glGetObjectParameterivARB(hProg, 
        GL_OBJECT_LINK_STATUS_ARB, &status));

    return (status != GL_FALSE);
}
