/*
 * GLSLGeometryShader.cpp
 *
 * Copyright (C) 2006 - 2007 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#ifdef _WIN32
#include <Windows.h>
#endif /* _WIN32 */

#include "vislib_gl/graphics/gl/GLSLGeometryShader.h"

#include "vislib/Array.h"
#include "vislib/String.h"
#include "vislib/memutils.h"
#include "vislib/sys/sysfunctions.h"
#include "vislib_gl/graphics/gl/glverify.h"


/*
 * vislib_gl::graphics::gl::GLSLGeometryShader::RequiredExtensions
 */
const char* vislib_gl::graphics::gl::GLSLGeometryShader::RequiredExtensions(void) {
    static vislib::StringA exts = vislib::StringA(vislib_gl::graphics::gl::GLSLShader::RequiredExtensions()) +
                                  " GL_EXT_gpu_shader4 GL_EXT_geometry_shader4 ";
    return exts.PeekBuffer();
}


/*
 * vislib_gl::graphics::gl::GLSLGeometryShader::GPU4_EXTENSION_DIRECTIVE
 */
const char* vislib_gl::graphics::gl::GLSLGeometryShader::GPU4_EXTENSION_DIRECTIVE =
    "#extension GL_EXT_gpu_shader4:enable\n";


/*
 * vislib_gl::graphics::gl::GLSLGeometryShader::GLSLGeometryShader
 */
vislib_gl::graphics::gl::GLSLGeometryShader::GLSLGeometryShader(void) : GLSLShader() {}


/*
 * vislib_gl::graphics::gl::GLSLGeometryShader::~GLSLGeometryShader
 */
vislib_gl::graphics::gl::GLSLGeometryShader::~GLSLGeometryShader(void) {}


/*
 * vislib_gl::graphics::gl::GLSLGeometryShader::Compile
 */
bool vislib_gl::graphics::gl::GLSLGeometryShader::Compile(
    const char* vertexShaderSrc, const char* geometryShaderSrc, const char* fragmentShaderSrc) {
    const char* v[] = {vertexShaderSrc};
    const char* g[] = {geometryShaderSrc};
    const char* f[] = {fragmentShaderSrc};

    return this->Compile(v, 1, g, 1, f, 1);
}


/*
 * vislib_gl::graphics::gl::GLSLGeometryShader::Compile
 */
bool vislib_gl::graphics::gl::GLSLGeometryShader::Compile(const char** vertexShaderSrc, const SIZE_T cntVertexShaderSrc,
    const char** geometryShaderSrc, const SIZE_T cntGeometryShaderSrc, const char** fragmentShaderSrc,
    const SIZE_T cntFragmentShaderSrc, bool insertLineDirective) {

    USES_GL_VERIFY;
    ASSERT(vertexShaderSrc != NULL);
    ASSERT(geometryShaderSrc != NULL);
    ASSERT(fragmentShaderSrc != NULL);

    this->Release();

    GLhandleARB hPixelShader = this->compileNewShader(
        GL_FRAGMENT_SHADER_ARB, fragmentShaderSrc, static_cast<GLsizei>(cntFragmentShaderSrc), insertLineDirective);
    GLhandleARB hGeometryShader = this->compileNewShader(
        GL_GEOMETRY_SHADER_EXT, geometryShaderSrc, static_cast<GLsizei>(cntGeometryShaderSrc), insertLineDirective);
    GLhandleARB hVertexShader = this->compileNewShader(
        GL_VERTEX_SHADER_ARB, vertexShaderSrc, static_cast<GLsizei>(cntVertexShaderSrc), insertLineDirective);

    /* Assemble program object. */
    GL_VERIFY_THROW(this->hProgObj = ::glCreateProgram());
    GL_VERIFY_THROW(::glAttachShader(this->hProgObj, hVertexShader));
    GL_VERIFY_THROW(::glAttachShader(this->hProgObj, hGeometryShader));
    GL_VERIFY_THROW(::glAttachShader(this->hProgObj, hPixelShader));

    return true;
}


/*
 * vislib_gl::graphics::gl::GLSLGeometryShader::CompileFromFile
 */
bool vislib_gl::graphics::gl::GLSLGeometryShader::CompileFromFile(
    const char* vertexShaderFile, const char* geometryShaderFile, const char* fragmentShaderFile) {
    vislib::StringA vertexShaderSrc;
    vislib::StringA geometryShaderSrc;
    vislib::StringA fragmentShaderSrc;

    if (!vislib::sys::ReadTextFile(vertexShaderSrc, vertexShaderFile)) {
        return false;
    }

    if (!vislib::sys::ReadTextFile(geometryShaderSrc, geometryShaderFile)) {
        return false;
    }

    if (!vislib::sys::ReadTextFile(fragmentShaderSrc, fragmentShaderFile)) {
        return false;
    }

    return this->Compile(vertexShaderSrc, geometryShaderSrc, fragmentShaderSrc);
}


/*
 * vislib_gl::graphics::gl::GLSLGeometryShader::CompileFromFile
 */
bool vislib_gl::graphics::gl::GLSLGeometryShader::CompileFromFile(const char** vertexShaderFiles,
    const SIZE_T cntVertexShaderFiles, const char** geometryShaderFiles, const SIZE_T cntGeometryShaderFiles,
    const char** fragmentShaderFiles, const SIZE_T cntFragmentShaderFiles, bool insertLineDirective) {

    // using arrays for automatic cleanup when a 'read' throws an exception
    vislib::Array<vislib::StringA> vertexShaderSrcs(cntVertexShaderFiles);
    vislib::Array<vislib::StringA> geometryShaderSrcs(cntGeometryShaderFiles);
    vislib::Array<vislib::StringA> fragmentShaderSrcs(cntFragmentShaderFiles);

    for (SIZE_T i = 0; i < cntVertexShaderFiles; i++) {
        if (!vislib::sys::ReadTextFile(vertexShaderSrcs[i], vertexShaderFiles[i])) {
            return false;
        }
    }

    for (SIZE_T i = 0; i < cntGeometryShaderFiles; i++) {
        if (!vislib::sys::ReadTextFile(geometryShaderSrcs[i], geometryShaderFiles[i])) {
            return false;
        }
    }

    for (SIZE_T i = 0; i < cntFragmentShaderFiles; i++) {
        if (!vislib::sys::ReadTextFile(fragmentShaderSrcs[i], fragmentShaderFiles[i])) {
            return false;
        }
    }

    // built up pointer arrays for attributes
    const char** vertexShaderSrcPtrs = new const char*[cntVertexShaderFiles];
    const char** geometryShaderSrcPtrs = new const char*[cntGeometryShaderFiles];
    const char** fragmentShaderSrcPtrs = new const char*[cntFragmentShaderFiles];

    try {
        for (SIZE_T i = 0; i < cntVertexShaderFiles; i++) {
            vertexShaderSrcPtrs[i] = vertexShaderSrcs[i].PeekBuffer();
        }
        for (SIZE_T i = 0; i < cntGeometryShaderFiles; i++) {
            geometryShaderSrcPtrs[i] = geometryShaderSrcs[i].PeekBuffer();
        }
        for (SIZE_T i = 0; i < cntFragmentShaderFiles; i++) {
            fragmentShaderSrcPtrs[i] = fragmentShaderSrcs[i].PeekBuffer();
        }

        bool retval = this->Compile(vertexShaderSrcPtrs, cntVertexShaderFiles, geometryShaderSrcPtrs,
            cntGeometryShaderFiles, fragmentShaderSrcPtrs, cntFragmentShaderFiles, insertLineDirective);

        ARY_SAFE_DELETE(vertexShaderSrcPtrs);
        ARY_SAFE_DELETE(geometryShaderSrcPtrs);
        ARY_SAFE_DELETE(fragmentShaderSrcPtrs);

        return retval;

        // free pointer arrays on exception
    } catch (OpenGLException e) { // catch OpenGLException to avoid truncating
        ARY_SAFE_DELETE(vertexShaderSrcPtrs);
        ARY_SAFE_DELETE(geometryShaderSrcPtrs);
        ARY_SAFE_DELETE(fragmentShaderSrcPtrs);
        throw e;
    } catch (CompileException e) {
        ARY_SAFE_DELETE(vertexShaderSrcPtrs);
        ARY_SAFE_DELETE(geometryShaderSrcPtrs);
        ARY_SAFE_DELETE(fragmentShaderSrcPtrs);
        throw e;
    } catch (vislib::Exception e) {
        ARY_SAFE_DELETE(vertexShaderSrcPtrs);
        ARY_SAFE_DELETE(geometryShaderSrcPtrs);
        ARY_SAFE_DELETE(fragmentShaderSrcPtrs);
        throw e;
    } catch (...) {
        ARY_SAFE_DELETE(vertexShaderSrcPtrs);
        ARY_SAFE_DELETE(geometryShaderSrcPtrs);
        ARY_SAFE_DELETE(fragmentShaderSrcPtrs);
        throw vislib::Exception("Unknown Exception", __FILE__, __LINE__);
    }

    return false; // should be unreachable code!
}


/*
 * vislib_gl::graphics::gl::GLSLGeometryShader::SetProgramParameter
 */
void vislib_gl::graphics::gl::GLSLGeometryShader::SetProgramParameter(GLenum name, GLint value) {
    ASSERT(GLSLShader::IsValidHandle(this->hProgObj));

    glProgramParameteriEXT(this->hProgObj, name, value);
}
