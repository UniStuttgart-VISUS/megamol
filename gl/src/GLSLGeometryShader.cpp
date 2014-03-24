/*
 * GLSLGeometryShader.cpp
 *
 * Copyright (C) 2006 - 2007 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#ifdef _WIN32
#include <windows.h>
#endif /* _WIN32 */

#include "vislib/GLSLGeometryShader.h"

#include "vislib/Array.h"
#include "vislib/glverify.h"
#include "the/memory.h"
#include "vislib/String.h"
#include "vislib/sysfunctions.h"


/*
 * vislib::graphics::gl::GLSLGeometryShader::RequiredExtensions
 */
const char * 
vislib::graphics::gl::GLSLGeometryShader::RequiredExtensions(void) {
    static vislib::StringA exts = vislib::StringA(
        vislib::graphics::gl::GLSLShader::RequiredExtensions())
        + " GL_EXT_gpu_shader4 GL_EXT_geometry_shader4 ";
    return exts.PeekBuffer();
}


/*
 * vislib::graphics::gl::GLSLGeometryShader::GPU4_EXTENSION_DIRECTIVE
 */
const char *
vislib::graphics::gl::GLSLGeometryShader::GPU4_EXTENSION_DIRECTIVE =
    "#extension GL_EXT_gpu_shader4:enable\n";


/*
 * vislib::graphics::gl::GLSLGeometryShader::GLSLGeometryShader
 */
vislib::graphics::gl::GLSLGeometryShader::GLSLGeometryShader(void) 
        : GLSLShader() {
}


/*
 * vislib::graphics::gl::GLSLGeometryShader::~GLSLGeometryShader
 */
vislib::graphics::gl::GLSLGeometryShader::~GLSLGeometryShader(void) {
}


/*
 * vislib::graphics::gl::GLSLGeometryShader::Compile
 */
bool vislib::graphics::gl::GLSLGeometryShader::Compile(
        const char *vertexShaderSrc, const char *geometryShaderSrc, 
        const char *fragmentShaderSrc) {
    const char *v[] = { vertexShaderSrc };
    const char *g[] = { geometryShaderSrc };
    const char *f[] = { fragmentShaderSrc };
    
    return this->Compile(v, 1, g, 1, f, 1);
}


/*
 * vislib::graphics::gl::GLSLGeometryShader::Compile
 */
bool vislib::graphics::gl::GLSLGeometryShader::Compile(
        const char **vertexShaderSrc, const size_t cntVertexShaderSrc, 
        const char **geometryShaderSrc, const size_t cntGeometryShaderSrc, 
        const char **fragmentShaderSrc, const size_t cntFragmentShaderSrc, 
        bool insertLineDirective) {

    USES_GL_VERIFY;
    THE_ASSERT(vertexShaderSrc != NULL);
    THE_ASSERT(geometryShaderSrc != NULL);
    THE_ASSERT(fragmentShaderSrc != NULL);

    this->Release();

    GLhandleARB hPixelShader = this->compileNewShader(GL_FRAGMENT_SHADER_ARB,
        fragmentShaderSrc, static_cast<GLsizei>(cntFragmentShaderSrc), 
        insertLineDirective);
    GLhandleARB hGeometryShader = this->compileNewShader(GL_GEOMETRY_SHADER_EXT,
        geometryShaderSrc, static_cast<GLsizei>(cntGeometryShaderSrc), 
        insertLineDirective);
    GLhandleARB hVertexShader = this->compileNewShader(GL_VERTEX_SHADER_ARB,
        vertexShaderSrc, static_cast<GLsizei>(cntVertexShaderSrc), 
        insertLineDirective);

    /* Assemble program object. */
    GL_VERIFY_THROW(this->hProgObj = ::glCreateProgramObjectARB());
    GL_VERIFY_THROW(::glAttachObjectARB(this->hProgObj, hVertexShader));
    GL_VERIFY_THROW(::glAttachObjectARB(this->hProgObj, hGeometryShader));
    GL_VERIFY_THROW(::glAttachObjectARB(this->hProgObj, hPixelShader));

    return true;
}


/*
 * vislib::graphics::gl::GLSLGeometryShader::CompileFromFile
 */
bool vislib::graphics::gl::GLSLGeometryShader::CompileFromFile(
        const char *vertexShaderFile, const char *geometryShaderFile, 
        const char *fragmentShaderFile) {
    StringA vertexShaderSrc;
    StringA geometryShaderSrc;
    StringA fragmentShaderSrc;

    if (!vislib::sys::ReadTextFile(vertexShaderSrc, vertexShaderFile)) {
        return false;
    }

    if (!vislib::sys::ReadTextFile(geometryShaderSrc, geometryShaderFile)) {
        return false;
    }

    if (!vislib::sys::ReadTextFile(fragmentShaderSrc, fragmentShaderFile)) {
        return false;
    }

    return this->Compile(vertexShaderSrc, geometryShaderSrc, 
        fragmentShaderSrc);
}


/*
 * vislib::graphics::gl::GLSLGeometryShader::CompileFromFile
 */
bool vislib::graphics::gl::GLSLGeometryShader::CompileFromFile(
        const char **vertexShaderFiles, const size_t cntVertexShaderFiles,
        const char **geometryShaderFiles, const size_t cntGeometryShaderFiles,
        const char **fragmentShaderFiles, const size_t cntFragmentShaderFiles,
        bool insertLineDirective) {

    // using arrays for automatic cleanup when a 'read' throws an exception
    Array<StringA> vertexShaderSrcs(cntVertexShaderFiles);
    Array<StringA> geometryShaderSrcs(cntGeometryShaderFiles);
    Array<StringA> fragmentShaderSrcs(cntFragmentShaderFiles);

    for(size_t i = 0; i < cntVertexShaderFiles; i++) {
        if (!vislib::sys::ReadTextFile(vertexShaderSrcs[i], 
                vertexShaderFiles[i])) {
            return false;
        }
    }

    for(size_t i = 0; i < cntGeometryShaderFiles; i++) {
        if (!vislib::sys::ReadTextFile(geometryShaderSrcs[i], 
                geometryShaderFiles[i])) {
            return false;
        }
    }

    for(size_t i = 0; i < cntFragmentShaderFiles; i++) {
        if (!vislib::sys::ReadTextFile(fragmentShaderSrcs[i], 
                fragmentShaderFiles[i])) {
            return false;
        }
    }

    // built up pointer arrays for attributes
    const char **vertexShaderSrcPtrs = new const char*[cntVertexShaderFiles];
    const char **geometryShaderSrcPtrs 
        = new const char*[cntGeometryShaderFiles];
    const char **fragmentShaderSrcPtrs 
        = new const char*[cntFragmentShaderFiles];

    try {
        for(size_t i = 0; i < cntVertexShaderFiles; i++) {
            vertexShaderSrcPtrs[i] = vertexShaderSrcs[i].PeekBuffer();
        }
        for(size_t i = 0; i < cntGeometryShaderFiles; i++) {
            geometryShaderSrcPtrs[i] = geometryShaderSrcs[i].PeekBuffer();
        }
        for(size_t i = 0; i < cntFragmentShaderFiles; i++) {
            fragmentShaderSrcPtrs[i] = fragmentShaderSrcs[i].PeekBuffer();
        }

        bool retval = this->Compile(vertexShaderSrcPtrs, cntVertexShaderFiles,
            geometryShaderSrcPtrs, cntGeometryShaderFiles, 
            fragmentShaderSrcPtrs, cntFragmentShaderFiles, 
            insertLineDirective);

        the::safe_array_delete(vertexShaderSrcPtrs);
        the::safe_array_delete(geometryShaderSrcPtrs);
        the::safe_array_delete(fragmentShaderSrcPtrs);

        return retval;

        // free pointer arrays on exception
    } catch(OpenGLException e) { // catch OpenGLException to avoid truncating
        the::safe_array_delete(vertexShaderSrcPtrs);
        the::safe_array_delete(geometryShaderSrcPtrs);
        the::safe_array_delete(fragmentShaderSrcPtrs);
        throw e;
    } catch(CompileException e) {
        the::safe_array_delete(vertexShaderSrcPtrs);
        the::safe_array_delete(geometryShaderSrcPtrs);
        the::safe_array_delete(fragmentShaderSrcPtrs);
        throw e;
    } catch(the::exception e) {
        the::safe_array_delete(vertexShaderSrcPtrs);
        the::safe_array_delete(geometryShaderSrcPtrs);
        the::safe_array_delete(fragmentShaderSrcPtrs);
        throw e;
    } catch(...) {
        the::safe_array_delete(vertexShaderSrcPtrs);
        the::safe_array_delete(geometryShaderSrcPtrs);
        the::safe_array_delete(fragmentShaderSrcPtrs);
        throw the::exception("Unknown Exception", __FILE__, __LINE__);
    }

    return false; // should be unreachable code!
}


/*
 * vislib::graphics::gl::GLSLGeometryShader::SetProgramParameter
 */
void vislib::graphics::gl::GLSLGeometryShader::SetProgramParameter(GLenum name,
        GLint value) {
    THE_ASSERT(GLSLShader::IsValidHandle(this->hProgObj));

    glProgramParameteriEXT(this->hProgObj, name, value);
}
