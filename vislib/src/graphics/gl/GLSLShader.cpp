/*
 * GLSLShader.cpp
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */

#ifdef _WIN32
#include <Windows.h>
#endif /* _WIN32 */

#include "vislib/graphics/gl/GLSLShader.h"

#include "vislib/Array.h"
#include "vislib/IllegalParamException.h"
#include "vislib/IllegalStateException.h"
#include "vislib/memutils.h"
#include "vislib/RawStorage.h"
#include "vislib/sys/sysfunctions.h"
#include "vislib/UnsupportedOperationException.h"


/*
 * vislib::graphics::gl::GLSLShader::IsValidHandle
 */
bool vislib::graphics::gl::GLSLShader::IsValidHandle(GLhandleARB hProg) {
    USES_GL_VERIFY;
    GLint status = 0;
    if (hProg == 0) return false;
    if (::glGetProgramiv != NULL) {
        if (GL_SUCCEEDED(::glGetProgramiv(hProg,
                GL_DELETE_STATUS, &status))) {
            return (status == 0);
        }
    }
    return false;
}


/*
 * vislib::graphics::gl::GLSLShader::RequiredExtensions
 */
const char * vislib::graphics::gl::GLSLShader::RequiredExtensions(void) {
    return "GL_ARB_shading_language_100 ";
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
 * vislib::graphics::gl::GLSLShader::BindAttribute
 */
GLenum vislib::graphics::gl::GLSLShader::BindAttribute(GLint index, 
        const char *name) {
    USES_GL_VERIFY;
    ASSERT(GLSLShader::IsValidHandle(this->hProgObj));

    GL_VERIFY_RETURN(::glBindAttribLocationARB(this->hProgObj, index, name));
    return GL_NO_ERROR;
}


/*
 * vislib::graphics::gl::GLSLShader::Compile
 */
bool vislib::graphics::gl::GLSLShader::Compile(const char *vertexShaderSrc, 
                                              const char *fragmentShaderSrc) {
    const char *v[] = { vertexShaderSrc };
    const char *f[] = { fragmentShaderSrc };
    
    return this->Compile(v, 1, f, 1, false);
}


/*
 * vislib::graphics::gl::GLSLShader::Compile
 */
bool vislib::graphics::gl::GLSLShader::Compile(const char **vertexShaderSrc, 
        const SIZE_T cntVertexShaderSrc, const char **fragmentShaderSrc,
        const SIZE_T cntFragmentShaderSrc, bool insertLineDirective) {
    USES_GL_VERIFY;
    ASSERT(vertexShaderSrc != NULL);
    ASSERT(fragmentShaderSrc != NULL);

    this->Release();

    GLhandleARB hPixelShader = this->compileNewShader(GL_FRAGMENT_SHADER_ARB,
        fragmentShaderSrc, static_cast<GLsizei>(cntFragmentShaderSrc), 
        insertLineDirective);
    GLhandleARB hVertexShader = this->compileNewShader(GL_VERTEX_SHADER_ARB,
        vertexShaderSrc, static_cast<GLsizei>(cntVertexShaderSrc), 
        insertLineDirective);

    /* Assemble program object. */
    GL_VERIFY_THROW(this->hProgObj = ::glCreateProgram());
    GL_VERIFY_THROW(::glAttachShader(this->hProgObj, hVertexShader));
    GL_VERIFY_THROW(::glAttachShader(this->hProgObj, hPixelShader));

    return true;
}


/*
 * vislib::graphics::gl::GLSLShader::CompileFromFile
 */
bool vislib::graphics::gl::GLSLShader::CompileFromFile(
        const char *vertexShaderFile, const char *fragmentShaderFile) {
    StringA vertexShaderSrc;
    StringA fragmentShaderSrc;

    if (!vislib::sys::ReadTextFile(vertexShaderSrc, vertexShaderFile)) {
        return false;
    }

    if (!vislib::sys::ReadTextFile(fragmentShaderSrc, fragmentShaderFile)) {
        return false;
    }

    return this->Compile(vertexShaderSrc, fragmentShaderSrc);
}


/*
 * vislib::graphics::gl::GLSLShader::CompileFromFile
 */
bool vislib::graphics::gl::GLSLShader::CompileFromFile(
        const char **vertexShaderFiles, const SIZE_T cntVertexShaderFiles, 
        const char **fragmentShaderFiles, 
        const SIZE_T cntFragmentShaderFiles, bool insertLineDirective) {

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

        bool retval = this->Compile(vertexShaderSrcPtrs, cntVertexShaderFiles, 
            fragmentShaderSrcPtrs, cntFragmentShaderFiles, 
            insertLineDirective);

        ARY_SAFE_DELETE(vertexShaderSrcPtrs);
        ARY_SAFE_DELETE(fragmentShaderSrcPtrs);

        return retval;

        // free pointer arrays on exception
    } catch(OpenGLException e) { // catch OpenGLException to avoid truncating
        ARY_SAFE_DELETE(vertexShaderSrcPtrs);
        ARY_SAFE_DELETE(fragmentShaderSrcPtrs);
        throw e;
    } catch(CompileException e) {
        ARY_SAFE_DELETE(vertexShaderSrcPtrs);
        ARY_SAFE_DELETE(fragmentShaderSrcPtrs);
        throw e;
    } catch(Exception e) {
        ARY_SAFE_DELETE(vertexShaderSrcPtrs);
        ARY_SAFE_DELETE(fragmentShaderSrcPtrs);
        throw e;
    } catch(...) {
        ARY_SAFE_DELETE(vertexShaderSrcPtrs);
        ARY_SAFE_DELETE(fragmentShaderSrcPtrs);
        throw Exception("Unknown Exception", __FILE__, __LINE__);
    }

    return false; // should be unreachable code!
}


/*
 * vislib::graphics::gl::GLSLShader::Create
 */
bool vislib::graphics::gl::GLSLShader::Create(const char *vertexShaderSrc, 
                                              const char *fragmentShaderSrc) {
    if (this->Compile(vertexShaderSrc, fragmentShaderSrc)) {
        return this->Link();
    } else {
        return false;
    }
}


/*
 * vislib::graphics::gl::GLSLShader::Create
 */
bool vislib::graphics::gl::GLSLShader::Create(const char **vertexShaderSrc, 
        const SIZE_T cntVertexShaderSrc, const char **fragmentShaderSrc,
        const SIZE_T cntFragmentShaderSrc, bool insertLineDirective) {
    if (this->Compile(vertexShaderSrc, cntVertexShaderSrc, fragmentShaderSrc, 
            cntFragmentShaderSrc, insertLineDirective)) {
        return this->Link();
    } else {
        return false;
    }
}


/*
 * vislib::graphics::gl::GLSLShader::CreateFromFile
 */
bool vislib::graphics::gl::GLSLShader::CreateFromFile(
        const char *vertexShaderFile, const char *fragmentShaderFile) {
    if (this->CompileFromFile(vertexShaderFile, fragmentShaderFile)) {
        return this->Link();
    } else {
        return false;
    }
}


/*
 * vislib::graphics::gl::GLSLShader::CreateFromFile
 */
bool vislib::graphics::gl::GLSLShader::CreateFromFile(
        const char **vertexShaderFiles, const SIZE_T cntVertexShaderFiles, 
        const char **fragmentShaderFiles, 
        const SIZE_T cntFragmentShaderFiles, bool insertLineDirective) {

    if (this->CompileFromFile(vertexShaderFiles, cntVertexShaderFiles, 
            fragmentShaderFiles, cntFragmentShaderFiles, 
            insertLineDirective)) {
        return this->Link();
    } else {
        return false;
    }
}


/*
 * vislib::graphics::gl::GLSLShader::Disable
 */
GLenum vislib::graphics::gl::GLSLShader::Disable(void) {
    USES_GL_VERIFY;
    ASSERT(GLSLShader::IsValidHandle(this->hProgObj));

    GL_VERIFY_RETURN(::glUseProgram(0));
    //GL_VERIFY_RETURN(::glDisable(GL_VERTEX_PROGRAM_ARB));
    //GL_VERIFY_RETURN(::glDisable(GL_FRAGMENT_PROGRAM_ARB));
    return GL_NO_ERROR;
}
        

/*
 * vislib::graphics::gl::GLSLShader::Enable
 */
GLenum vislib::graphics::gl::GLSLShader::Enable(void) {
    USES_GL_VERIFY;
    ASSERT(GLSLShader::IsValidHandle(this->hProgObj));

    //GL_VERIFY_RETURN(::glEnable(GL_VERTEX_PROGRAM_ARB));
    //GL_VERIFY_RETURN(::glEnable(GL_FRAGMENT_PROGRAM_ARB));
    GL_VERIFY_RETURN(::glUseProgram(this->hProgObj));
    return GL_NO_ERROR;
}


/*
 * vislib::graphics::gl::GLSLShader::Link
 */
bool vislib::graphics::gl::GLSLShader::Link() {
    USES_GL_VERIFY;
    ASSERT(GLSLShader::IsValidHandle(this->hProgObj));
    
    GL_VERIFY_THROW(::glLinkProgram(this->hProgObj));
    if (!this->isLinked(this->hProgObj)) {
        throw CompileException(this->getProgramInfoLog(this->hProgObj), 
            CompileException::ACTION_LINK, __FILE__, __LINE__);
    }

    return true;
}


/*
 * vislib::graphics::gl::GLSLShader::ParameterLocation
 */
GLint vislib::graphics::gl::GLSLShader::ParameterLocation(const char *name) const {
    ASSERT(name != NULL);
    ASSERT(GLSLShader::IsValidHandle(this->hProgObj));
    return ::glGetUniformLocation(this->hProgObj, name);
}


/*
 * vislib::graphics::gl::GLSLShader::Release
 */
GLenum vislib::graphics::gl::GLSLShader::Release(void) {
    USES_GL_VERIFY;

    if (GLSLShader::IsValidHandle(this->hProgObj)) {
        GLint objCnt;

        if (GL_SUCCEEDED(::glGetProgramiv(this->hProgObj, 
            GL_ATTACHED_SHADERS, &objCnt))) {
                        if (objCnt > 0) {
                                GLhandleARB *objs = new GLhandleARB[objCnt];

                                if (GL_SUCCEEDED(::glGetAttachedShaders(this->hProgObj,
                                        objCnt, &objCnt, objs))) {
                                        for (GLint i = 0; i < objCnt; i++) {
                                                ::glDetachShader(this->hProgObj, objs[i]);
                                                ::glDeleteShader(objs[i]);
                                        }
                                }
                                delete[] objs;
                        }
        }

        GL_VERIFY_RETURN(::glDeleteProgram(this->hProgObj));
    }

    return GL_NO_ERROR;
}


/*
 * vislib::graphics::gl::GLSLShader::SetParameter
 */
GLenum vislib::graphics::gl::GLSLShader::SetParameter(const GLint name,
        const float v1) {
    USES_GL_VERIFY;
    ASSERT(GLSLShader::IsValidHandle(this->hProgObj));

    if (name < 0) {
        return GL_INVALID_VALUE;
    }
    GL_VERIFY_RETURN(::glUniform1f(name, v1));
    return GL_NO_ERROR;

}


/*
 * vislib::graphics::gl::GLSLShader::SetParameter
 */
GLenum vislib::graphics::gl::GLSLShader::SetParameter(const GLint name, 
        const float v1, const float v2) {
    USES_GL_VERIFY;
    ASSERT(GLSLShader::IsValidHandle(this->hProgObj));

    if (name < 0) {
        return GL_INVALID_VALUE;
    }
    GL_VERIFY_RETURN(::glUniform2f(name, v1, v2));
    return GL_NO_ERROR;
}


/*
 * vislib::graphics::gl::GLSLShader::SetParameter
 */
GLenum vislib::graphics::gl::GLSLShader::SetParameter(const GLint name, 
        const float v1, const float v2, const float v3) {
    USES_GL_VERIFY;
    ASSERT(GLSLShader::IsValidHandle(this->hProgObj));

    if (name < 0) {
        return GL_INVALID_VALUE;
    }
    GL_VERIFY_RETURN(::glUniform3f(name, v1, v2, v3));
    return GL_NO_ERROR;
}


/*
 * vislib::graphics::gl::GLSLShader::SetParameter
 */
GLenum vislib::graphics::gl::GLSLShader::SetParameter(const GLint name, 
        const float v1, const float v2, const float v3, const float v4) {
    USES_GL_VERIFY;
    ASSERT(GLSLShader::IsValidHandle(this->hProgObj));

    if (name < 0) {
        return GL_INVALID_VALUE;
    }
    GL_VERIFY_RETURN(::glUniform4f(name, v1, v2, v3, v4));
    return GL_NO_ERROR;
}


/*
 * vislib::graphics::gl::GLSLShader::SetParameter
 */
GLenum vislib::graphics::gl::GLSLShader::SetParameter(const GLint name, 
        const int v1) {
    USES_GL_VERIFY;
    ASSERT(GLSLShader::IsValidHandle(this->hProgObj));

    if (name < 0) {
        return GL_INVALID_VALUE;
    }
    GL_VERIFY_RETURN(::glUniform1i(name, v1));
    return GL_NO_ERROR;
}


/*
 * vislib::graphics::gl::GLSLShader::SetParameter
 */
GLenum vislib::graphics::gl::GLSLShader::SetParameter(const GLint name, 
        const int v1, const int v2) {
    USES_GL_VERIFY;
    ASSERT(GLSLShader::IsValidHandle(this->hProgObj));

    if (name < 0) {
        return GL_INVALID_VALUE;
    }
    GL_VERIFY_RETURN(::glUniform2i(name, v1, v2));
    return GL_NO_ERROR;
}


/*
 * vislib::graphics::gl::GLSLShader::SetParameter
 */
GLenum vislib::graphics::gl::GLSLShader::SetParameter(const GLint name, 
        const int v1, const int v2, const int v3) {
    USES_GL_VERIFY;
    ASSERT(GLSLShader::IsValidHandle(this->hProgObj));

    if (name < 0) {
        return GL_INVALID_VALUE;
    }
    GL_VERIFY_RETURN(::glUniform3i(name, v1, v2, v3));
    return GL_NO_ERROR;
}


/*
 * vislib::graphics::gl::GLSLShader::SetParameter
 */
GLenum vislib::graphics::gl::GLSLShader::SetParameter(const GLint name, 
        const int v1, const int v2, const int v3, const int v4) {
    USES_GL_VERIFY;
    ASSERT(GLSLShader::IsValidHandle(this->hProgObj));

    if (name < 0) {
        return GL_INVALID_VALUE;
    }
    GL_VERIFY_RETURN(::glUniform4i(name, v1, v2, v3, v4));
    return GL_NO_ERROR;
}


/*
 * vislib::graphics::gl::GLSLShader::SetParameterArray1
 */
GLenum vislib::graphics::gl::GLSLShader::SetParameterArray1(const GLint name,
        const GLsizei count, const float *value) {
    USES_GL_VERIFY;
    ASSERT(GLSLShader::IsValidHandle(this->hProgObj));

    if (name < 0) {
        return GL_INVALID_VALUE;
    }
    GL_VERIFY_RETURN(::glUniform1fv(name, count, value));
    return GL_NO_ERROR;
}


/*
 * vislib::graphics::gl::GLSLShader::SetParameterArray2
 */
GLenum vislib::graphics::gl::GLSLShader::SetParameterArray2(const GLint name,
        const GLsizei count, const float *value) {
    USES_GL_VERIFY;
    ASSERT(GLSLShader::IsValidHandle(this->hProgObj));

    if (name < 0) {
        return GL_INVALID_VALUE;
    }
    GL_VERIFY_RETURN(::glUniform2fv(name, count, value));
    return GL_NO_ERROR;
}


/*
 * vislib::graphics::gl::GLSLShader::SetParameterArray3
 */
GLenum vislib::graphics::gl::GLSLShader::SetParameterArray3(const GLint name,
        const GLsizei count, const float *value) {
    USES_GL_VERIFY;
    ASSERT(GLSLShader::IsValidHandle(this->hProgObj));

    if (name < 0) {
        return GL_INVALID_VALUE;
    }
    GL_VERIFY_RETURN(::glUniform3fv(name, count, value));
    return GL_NO_ERROR;
}


/*
 * vislib::graphics::gl::GLSLShader::SetParameterArray4
 */
GLenum vislib::graphics::gl::GLSLShader::SetParameterArray4(const GLint name,
        const GLsizei count, const float *value) {
    USES_GL_VERIFY;
    ASSERT(GLSLShader::IsValidHandle(this->hProgObj));

    if (name < 0) {
        return GL_INVALID_VALUE;
    }
    GL_VERIFY_RETURN(::glUniform4fv(name, count, value));
    return GL_NO_ERROR;
}


/*
 * vislib::graphics::gl::GLSLShader::SetParameterArray1
 */
GLenum vislib::graphics::gl::GLSLShader::SetParameterArray1(const GLint name,
        const GLsizei count, const int *value) {
    USES_GL_VERIFY;
    ASSERT(GLSLShader::IsValidHandle(this->hProgObj));

    if (name < 0) {
        return GL_INVALID_VALUE;
    }
    GL_VERIFY_RETURN(::glUniform1iv(name, count, value));
    return GL_NO_ERROR;
}


/*
 * vislib::graphics::gl::GLSLShader::SetParameterArray2
 */
GLenum vislib::graphics::gl::GLSLShader::SetParameterArray2(const GLint name,
        const GLsizei count, const int *value) {
    USES_GL_VERIFY;
    ASSERT(GLSLShader::IsValidHandle(this->hProgObj));

    if (name < 0) {
        return GL_INVALID_VALUE;
    }
    GL_VERIFY_RETURN(::glUniform2iv(name, count, value));
    return GL_NO_ERROR;
}


/*
 * vislib::graphics::gl::GLSLShader::SetParameterArray3
 */
GLenum vislib::graphics::gl::GLSLShader::SetParameterArray3(const GLint name,
        const GLsizei count, const int *value) {
    USES_GL_VERIFY;
    ASSERT(GLSLShader::IsValidHandle(this->hProgObj));

    if (name < 0) {
        return GL_INVALID_VALUE;
    }
    GL_VERIFY_RETURN(::glUniform3iv(name, count, value));
    return GL_NO_ERROR;
}


/*
 * vislib::graphics::gl::GLSLShader::SetParameterArray4
 */
GLenum vislib::graphics::gl::GLSLShader::SetParameterArray4(const GLint name,
        const GLsizei count, const int *value) {
    USES_GL_VERIFY;
    ASSERT(GLSLShader::IsValidHandle(this->hProgObj));

    if (name < 0) {
        return GL_INVALID_VALUE;
    }
    GL_VERIFY_RETURN(::glUniform4iv(name, count, value));
    return GL_NO_ERROR;
}


/*
 * vislib::graphics::gl::GLSLShader::compileNewShader
 */
GLhandleARB vislib::graphics::gl::GLSLShader::compileNewShader(GLenum type, 
        const char **src, GLsizei cnt, bool insertLineDirective) {
    USES_GL_VERIFY;
    GLhandleARB shader;
    RawStorage powerMemory;
    const char lineStr[] = "\n#line 0 %d\n";

    if (insertLineDirective && (cnt > 1)) {
        StringA tmp;
        char *ptr;
        tmp.Format(lineStr, cnt);

        // very tricky:
        powerMemory.AssertSize((sizeof(char*) * (cnt * 2 - 1)) 
            + ((cnt - 1) * sizeof(char) * (tmp.Length() + 1)));
        ptr = powerMemory.As<char>() + (sizeof(char*) * (cnt = cnt * 2 - 1));
        for (GLsizei i = 0; i < cnt; i++) {
            if (i % 2 == 0) {
                powerMemory.As<const char*>()[i] = src[i / 2];
            } else {
                unsigned int len;
                tmp.Format(lineStr, int((i + 1) / 2));
                len = (tmp.Length() + 1) * sizeof(char);
                memcpy(ptr, tmp.PeekBuffer(), len);
                powerMemory.As<char*>()[i] = ptr;
                ptr += len;
            }
        }

        src = powerMemory.As<const char*>();
    }

    GL_VERIFY_THROW(shader = ::glCreateShader(type));
    GL_VERIFY_THROW(::glShaderSource(shader, cnt, src, NULL));
    GL_VERIFY_THROW(::glCompileShader(shader));

    GLint status;
    GL_VERIFY_THROW(glGetShaderiv(shader, GL_COMPILE_STATUS, &status));
    if (status != GL_TRUE) {
#if 0
        printf("full shader:\n");
        for (int x = 0; x < cnt; x++) {
            printf("%s\n//snippet end\n", src[x]);
        }
        printf("\n\n");
#endif
        throw CompileException(getShaderInfoLog(shader),
            CompileException::CompilationFailedAction(type),
            __FILE__, __LINE__);
    }

    return shader;
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

    GL_VERIFY_THROW(::glGetProgramiv(hProg,
        GL_INFO_LOG_LENGTH, &len));

    if (len > 0) {
        log = retval.AllocateBuffer(len);
        GL_VERIFY_THROW(::glGetProgramInfoLog(hProg, len, &written, log));
    }

    return retval;
}


/*
 * vislib::graphics::gl::GLSLShader::getShaderInfoLog
 */
vislib::StringA vislib::graphics::gl::GLSLShader::getShaderInfoLog(
        GLhandleARB hShader) {
    USES_GL_VERIFY;
    GLint len = 0;
    GLint written = 0;
    StringA retval;
    char *log = NULL;

    GL_VERIFY_THROW(::glGetShaderiv(hShader, GL_INFO_LOG_LENGTH, &len));

    if (len > 0) {
        log = retval.AllocateBuffer(len);
        GL_VERIFY_THROW(::glGetShaderInfoLog(hShader, len, &written, log));
    }

    return retval;
}


/*
 * vislib::graphics::gl::GLSLShader::isCompiled
 */
bool vislib::graphics::gl::GLSLShader::isCompiled(GLhandleARB hProg) {
    USES_GL_VERIFY;
    GLint s, objCnt;
    bool status = true;

    GL_VERIFY_THROW(::glGetProgramiv(hProg, GL_ATTACHED_SHADERS, &objCnt));

    GLhandleARB *objs = new GLhandleARB[objCnt];

    if (GL_SUCCEEDED(::glGetAttachedShaders(hProg, objCnt, &objCnt, objs))) {
        for (GLint i = 0; i < objCnt; i++) {
            glGetShaderiv(objs[i], GL_COMPILE_STATUS, &s);
            status = status && (s == GL_TRUE);
        }
    }

    delete[] objs;

    return status;
}


/*
 * vislib::graphics::gl::GLSLShader::isLinked
 */
bool vislib::graphics::gl::GLSLShader::isLinked(GLhandleARB hProg) {
    USES_GL_VERIFY;
    GLint status;

    GL_VERIFY_THROW(::glGetProgramiv(hProg, 
        GL_LINK_STATUS, &status));

    return (status != GL_FALSE);
}
