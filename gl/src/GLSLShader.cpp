/*
 * GLSLShader.cpp
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */

#ifdef _WIN32
#include <windows.h>
#endif /* _WIN32 */

#include "vislib/GLSLShader.h"

#include "vislib/Array.h"
#include "vislib/glverify.h"
#include "vislib/IllegalParamException.h"
#include "vislib/IllegalStateException.h"
#include "vislib/sysfunctions.h"
#include "vislib/UnsupportedOperationException.h"


/*
 * vislib::graphics::gl::GLSLShader::IsValidHandle
 */
bool vislib::graphics::gl::GLSLShader::IsValidHandle(GLhandleARB hProg) {
    USES_GL_VERIFY;
    GLint status;

    if (GL_SUCCEEDED(::glGetObjectParameterivARB(hProg, 
            GL_OBJECT_DELETE_STATUS_ARB, &status))) {
        return (status == 0);
    } else {
        return false;
    }
}


/*
 * vislib::graphics::gl::GLSLShader::RequiredExtensions
 */
const char * vislib::graphics::gl::GLSLShader::RequiredExtensions(void) {
    return "GL_ARB_shader_objects ";
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

    if (!vislib::sys::ReadTextFile(vertexShaderSrc, vertexShaderFile)) {
        return false;
    }

    if (!vislib::sys::ReadTextFile(fragmentShaderSrc, fragmentShaderFile)) {
        return false;
    }

    return this->Create(vertexShaderSrc, fragmentShaderSrc);
}


/*
 * vislib::graphics::gl::GLSLShader::CreateFromFiles
 */
bool vislib::graphics::gl::GLSLShader::CreateFromFiles(
        const char **vertexShaderFiles, const SIZE_T cntVertexShaderFiles, 
        const char **fragmentShaderFiles, 
        const SIZE_T cntFragmentShaderFiles) {

    // using arrays for automatic cleanup when a 'read' throws an exception
    Array<StringA> vertexShaderSrcs(cntVertexShaderFiles);
    Array<StringA> fragmentShaderSrcs(cntFragmentShaderFiles);

    for(SIZE_T i = 0; i < cntVertexShaderFiles; i++) {
        if (!vislib::sys::ReadTextFile(vertexShaderSrcs[i], 
                vertexShaderFiles[i])) {
            return false;
        }
    }

    for(SIZE_T i = 0; i < cntFragmentShaderFiles; i++) {
        if (!vislib::sys::ReadTextFile(fragmentShaderSrcs[i], 
                fragmentShaderFiles[i])) {
            return false;
        }
    }

    // built up pointer arrays for attributes
    const char **vertexShaderSrcPtrs = new const char*[cntVertexShaderFiles];
    const char **fragmentShaderSrcPtrs 
        = new const char*[cntFragmentShaderFiles];

    try {
        for(SIZE_T i = 0; i < cntVertexShaderFiles; i++) {
            vertexShaderSrcPtrs[i] = vertexShaderSrcs[i].PeekBuffer();
        }
        for(SIZE_T i = 0; i < cntFragmentShaderFiles; i++) {
            fragmentShaderSrcPtrs[i] = fragmentShaderSrcs[i].PeekBuffer();
        }

        bool retval = this->Create(vertexShaderSrcPtrs, cntVertexShaderFiles, 
            fragmentShaderSrcPtrs, cntFragmentShaderFiles);

        delete[] vertexShaderSrcPtrs;
        delete[] fragmentShaderSrcPtrs;

        return retval;

        // free pointer arrays on exception
    } catch(OpenGLException e) { // catch OpenGLException to avoid truncating
        delete[] vertexShaderSrcPtrs;
        delete[] fragmentShaderSrcPtrs;
        throw e;
    } catch(CompileException e) {
        delete[] vertexShaderSrcPtrs;
        delete[] fragmentShaderSrcPtrs;
        throw e;
    } catch(Exception e) {
        delete[] vertexShaderSrcPtrs;
        delete[] fragmentShaderSrcPtrs;
        throw e;
    } catch(...) {
        delete[] vertexShaderSrcPtrs;
        delete[] fragmentShaderSrcPtrs;
        throw Exception("Unknown Exception", __FILE__, __LINE__);
    }

    return false; // should be unreachable code!
}


/*
 * vislib::graphics::gl::GLSLShader::Disable
 */
GLenum vislib::graphics::gl::GLSLShader::Disable(void) {
    USES_GL_VERIFY;
    ASSERT(GLSLShader::IsValidHandle(this->hProgObj));

    GL_VERIFY_RETURN(::glUseProgramObjectARB(0));
    GL_VERIFY_RETURN(::glDisable(GL_VERTEX_PROGRAM_ARB));
    GL_VERIFY_RETURN(::glDisable(GL_FRAGMENT_PROGRAM_ARB));
    return GL_NO_ERROR;
}
        

/*
 * vislib::graphics::gl::GLSLShader::Enable
 */
GLenum vislib::graphics::gl::GLSLShader::Enable(void) {
    USES_GL_VERIFY;
    ASSERT(GLSLShader::IsValidHandle(this->hProgObj));

    GL_VERIFY_RETURN(::glEnable(GL_VERTEX_PROGRAM_ARB));
    GL_VERIFY_RETURN(::glEnable(GL_FRAGMENT_PROGRAM_ARB));
    GL_VERIFY_RETURN(::glUseProgramObjectARB(this->hProgObj));
    return GL_NO_ERROR;
}


/*
 * vislib::graphics::gl::GLSLShader::ParameterLocation
 */
GLint vislib::graphics::gl::GLSLShader::ParameterLocation(const char *name) const {
    ASSERT(name != NULL);
    ASSERT(GLSLShader::IsValidHandle(this->hProgObj));
    return ::glGetUniformLocationARB(this->hProgObj, name);
}


/*
 * vislib::graphics::gl::GLSLShader::Release
 */
GLenum vislib::graphics::gl::GLSLShader::Release(void) {
    USES_GL_VERIFY;

    if (GLSLShader::IsValidHandle(this->hProgObj)) {
        GL_VERIFY_RETURN(::glDeleteObjectARB(this->hProgObj));
    }

    return GL_NO_ERROR;
}


/*
 * vislib::graphics::gl::GLSLShader::SetParameter
 */
GLenum vislib::graphics::gl::GLSLShader::SetParameter(const char *name,
                                                      const float v1) {
    USES_GL_VERIFY;
    ASSERT(name != NULL);
    ASSERT(GLSLShader::IsValidHandle(this->hProgObj));

    GLint location = ::glGetUniformLocationARB(this->hProgObj, name);

    if (location < 0) {
        return GL_INVALID_VALUE;
    }

    GL_VERIFY_RETURN(::glUniform1fARB(location, v1));
    return GL_NO_ERROR;

}


/*
 * vislib::graphics::gl::GLSLShader::SetParameter
 */
GLenum vislib::graphics::gl::GLSLShader::SetParameter(const char *name, 
        const float v1, const float v2) {
    USES_GL_VERIFY;
    ASSERT(name != NULL);
    ASSERT(GLSLShader::IsValidHandle(this->hProgObj));

    GLint location = ::glGetUniformLocationARB(this->hProgObj, name);

    if (location < 0) {
        return GL_INVALID_VALUE;
    }

    GL_VERIFY_RETURN(::glUniform2fARB(location, v1, v2));
    return GL_NO_ERROR;
}


/*
 * vislib::graphics::gl::GLSLShader::SetParameter
 */
GLenum vislib::graphics::gl::GLSLShader::SetParameter(const char *name, 
        const float v1, const float v2, const float v3) {
    USES_GL_VERIFY;
    ASSERT(name != NULL);
    ASSERT(GLSLShader::IsValidHandle(this->hProgObj));

    GLint location = ::glGetUniformLocationARB(this->hProgObj, name);

    if (location < 0) {
        return GL_INVALID_VALUE;
    }

    GL_VERIFY_RETURN(::glUniform3fARB(location, v1, v2, v3));
    return GL_NO_ERROR;
}


/*
 * vislib::graphics::gl::GLSLShader::SetParameter
 */
GLenum vislib::graphics::gl::GLSLShader::SetParameter(const char *name, 
        const float v1, const float v2, const float v3, const float v4) {
    USES_GL_VERIFY;
    ASSERT(name != NULL);
    ASSERT(GLSLShader::IsValidHandle(this->hProgObj));

    GLint location = ::glGetUniformLocationARB(this->hProgObj, name);

    if (location < 0) {
        return GL_INVALID_VALUE;
    }

    GL_VERIFY_RETURN(::glUniform4fARB(location, v1, v2, v3, v4));
    return GL_NO_ERROR;
}


/*
 * vislib::graphics::gl::GLSLShader::SetParameter
 */
GLenum vislib::graphics::gl::GLSLShader::SetParameter(const char *name, 
                                                      const int v1) {
    USES_GL_VERIFY;
    ASSERT(name != NULL);
    ASSERT(GLSLShader::IsValidHandle(this->hProgObj));

    GLint location = ::glGetUniformLocationARB(this->hProgObj, name);

    if (location < 0) {
        return GL_INVALID_VALUE;
    }

    GL_VERIFY_RETURN(::glUniform1iARB(location, v1));
    return GL_NO_ERROR;
}


/*
 * vislib::graphics::gl::GLSLShader::SetParameter
 */
GLenum vislib::graphics::gl::GLSLShader::SetParameter(const char *name, 
        const int v1, const int v2) {
    USES_GL_VERIFY;
    ASSERT(name != NULL);
    ASSERT(GLSLShader::IsValidHandle(this->hProgObj));

    GLint location = ::glGetUniformLocationARB(this->hProgObj, name);

    if (location < 0) {
        return GL_INVALID_VALUE;
    }

    GL_VERIFY_RETURN(::glUniform2iARB(location, v1, v2));
    return GL_NO_ERROR;
}


/*
 * vislib::graphics::gl::GLSLShader::SetParameter
 */
GLenum vislib::graphics::gl::GLSLShader::SetParameter(const char *name, 
        const int v1, const int v2, const int v3) {
    USES_GL_VERIFY;
    ASSERT(name != NULL);
    ASSERT(GLSLShader::IsValidHandle(this->hProgObj));

    GLint location = ::glGetUniformLocationARB(this->hProgObj, name);

    if (location < 0) {
        return GL_INVALID_VALUE;
    }

    GL_VERIFY_RETURN(::glUniform3iARB(location, v1, v2, v3));
    return GL_NO_ERROR;
}


/*
 * vislib::graphics::gl::GLSLShader::SetParameter
 */
GLenum vislib::graphics::gl::GLSLShader::SetParameter(const char *name, 
        const int v1, const int v2, const int v3, const int v4) {
    USES_GL_VERIFY;
    ASSERT(name != NULL);
    ASSERT(GLSLShader::IsValidHandle(this->hProgObj));

    GLint location = ::glGetUniformLocationARB(this->hProgObj, name);

    if (location < 0) {
        return GL_INVALID_VALUE;
    }

    GL_VERIFY_RETURN(::glUniform4iARB(location, v1, v2, v3, v4));
    return GL_NO_ERROR;
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
