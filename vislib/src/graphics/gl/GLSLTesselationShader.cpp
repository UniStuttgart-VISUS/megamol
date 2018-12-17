/*
 * GLSLGeometryShader.cpp
 *
 * Copyright (C) 2006 - 2007 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#ifdef _WIN32
#include <Windows.h>
#endif /* _WIN32 */

#include "vislib/graphics/gl/GLSLTesselationShader.h"

#include "vislib/Array.h"
#include "vislib/graphics/gl/glverify.h"
#include "vislib/memutils.h"
#include "vislib/String.h"
#include "vislib/sys/sysfunctions.h"


/*
 * vislib::graphics::gl::GLSLTesselationShader::RequiredExtensions
 */
const char * 
vislib::graphics::gl::GLSLTesselationShader::RequiredExtensions(void) {
    static vislib::StringA exts = vislib::StringA(
        vislib::graphics::gl::GLSLShader::RequiredExtensions())
        + " GL_ARB_gpu_shader5 GL_ARB_tessellation_shader ";
    return exts.PeekBuffer();
}


/*
 * vislib::graphics::gl::GLSLTesselationShader::GPU4_EXTENSION_DIRECTIVE
 */
const char *
vislib::graphics::gl::GLSLTesselationShader::GPU5_EXTENSION_DIRECTIVE =
    "#extension GL_EXT_gpu_shader5:enable\n";


/*
 * vislib::graphics::gl::GLSLTesselationShader::GLSLTesselationShader
 */
vislib::graphics::gl::GLSLTesselationShader::GLSLTesselationShader(void) 
        : GLSLShader() {
}


/*
 * vislib::graphics::gl::GLSLTesselationShader::~GLSLTesselationShader
 */
vislib::graphics::gl::GLSLTesselationShader::~GLSLTesselationShader(void) {
}


/*
 * vislib::graphics::gl::GLSLTesselationShader::Compile
 */
bool vislib::graphics::gl::GLSLTesselationShader::Compile(const char *vertexShaderSrc,
            const char *tessControlShaderSrc, const char *tessEvalShaderSrc,
            const char *geometryShaderSrc, const char *fragmentShaderSrc) {
    const char *v[] = { vertexShaderSrc };
    const char *tc[] = { tessControlShaderSrc };
    const char *te[] = { tessEvalShaderSrc };
    const char *g[] = { geometryShaderSrc };
    const char *f[] = { fragmentShaderSrc };
    
    return this->Compile(v, 1, tc, tessControlShaderSrc == NULL ? 0 : 1,
        te, tessEvalShaderSrc == NULL ? 0 : 1,
        g, geometryShaderSrc == NULL ? 0 : 1, f, 1);
}


/*
 * vislib::graphics::gl::GLSLTesselationShader::Compile
 */
bool vislib::graphics::gl::GLSLTesselationShader::Compile(
            const char **vertexShaderSrc, const SIZE_T cntVertexShaderSrc,
            const char **tessControlShaderSrc,
            const SIZE_T cntTessControlShaderSrc,
            const char **tessEvalShaderSrc, const SIZE_T cntTessEvalShaderSrc,
            const char **geometryShaderSrc, const SIZE_T cntGeometryShaderSrc,
            const char **fragmentShaderSrc, const SIZE_T cntFragmentShaderSrc,
            bool insertLineDirective) {

    USES_GL_VERIFY;
    ASSERT(vertexShaderSrc != NULL);
    //ASSERT(tessControlShaderSrc != NULL);
    //ASSERT(tessEvalShaderSrc != NULL);
    //ASSERT(geometryShaderSrc != NULL);
    ASSERT(fragmentShaderSrc != NULL);

    this->Release();

    GLhandleARB hTessControlShader;
    GLhandleARB hTessEvalShader;
    GLhandleARB hGeometryShader;

    GLhandleARB hPixelShader = this->compileNewShader(GL_FRAGMENT_SHADER_ARB,
        fragmentShaderSrc, static_cast<GLsizei>(cntFragmentShaderSrc), 
        insertLineDirective);
    if (cntTessControlShaderSrc != 0) {
        hTessControlShader = this->compileNewShader(GL_TESS_CONTROL_SHADER,
            tessControlShaderSrc, static_cast<GLsizei>(cntTessControlShaderSrc),
            insertLineDirective);
    }
    if (cntTessEvalShaderSrc != 0) {
        hTessEvalShader = this->compileNewShader(GL_TESS_EVALUATION_SHADER,
            tessEvalShaderSrc, static_cast<GLsizei>(cntTessEvalShaderSrc),
            insertLineDirective);
    }
    if (cntGeometryShaderSrc != 0) {
        hGeometryShader = this->compileNewShader(GL_GEOMETRY_SHADER_EXT,
            geometryShaderSrc, static_cast<GLsizei>(cntGeometryShaderSrc), 
            insertLineDirective);
    }
    GLhandleARB hVertexShader = this->compileNewShader(GL_VERTEX_SHADER_ARB,
        vertexShaderSrc, static_cast<GLsizei>(cntVertexShaderSrc), 
        insertLineDirective);

    /* Assemble program object. */
    GL_VERIFY_THROW(this->hProgObj = ::glCreateProgram());
    GL_VERIFY_THROW(::glAttachShader(this->hProgObj, hVertexShader));
    if (cntTessControlShaderSrc != 0) {
        GL_VERIFY_THROW(::glAttachShader(this->hProgObj, hTessControlShader));
    }
    if (cntTessEvalShaderSrc != 0) {
        GL_VERIFY_THROW(::glAttachShader(this->hProgObj, hTessEvalShader));
    }
    if (cntGeometryShaderSrc != 0) {
        GL_VERIFY_THROW(::glAttachShader(this->hProgObj, hGeometryShader));
    }
    GL_VERIFY_THROW(::glAttachShader(this->hProgObj, hPixelShader));

    return true;
}


/*
 * vislib::graphics::gl::GLSLTesselationShader::CompileFromFile
 */
bool vislib::graphics::gl::GLSLTesselationShader::CompileFromFile(
        const char *vertexShaderFile,
        const char *tessControlShaderFile, const char *tessEvalShaderFile,
        const char *geometryShaderFile, const char *fragmentShaderFile) {
    StringA vertexShaderSrc;
    StringA tessControlShaderSrc;
    StringA tessEvalShaderSrc;
    StringA geometryShaderSrc;
    StringA fragmentShaderSrc;

    if (!vislib::sys::ReadTextFile(vertexShaderSrc, vertexShaderFile)) {
        return false;
    }

    if (tessControlShaderFile != NULL && !vislib::sys::ReadTextFile(tessControlShaderSrc, tessControlShaderFile)) {
        return false;
    }

    if (tessEvalShaderFile != NULL && !vislib::sys::ReadTextFile(tessEvalShaderSrc, tessEvalShaderFile)) {
        return false;
    }

    if (geometryShaderFile != NULL && !vislib::sys::ReadTextFile(geometryShaderSrc, geometryShaderFile)) {
        return false;
    }

    if (!vislib::sys::ReadTextFile(fragmentShaderSrc, fragmentShaderFile)) {
        return false;
    }

    return this->Compile(vertexShaderSrc,
        tessControlShaderFile == NULL ? NULL : tessControlShaderSrc,
        tessEvalShaderFile == NULL ? NULL : tessEvalShaderSrc,
        geometryShaderFile == NULL ? NULL : geometryShaderSrc,
        fragmentShaderSrc);
}


/*
 * vislib::graphics::gl::GLSLTesselationShader::CompileFromFile
 */
bool vislib::graphics::gl::GLSLTesselationShader::CompileFromFile(
        const char **vertexShaderFiles,
        const SIZE_T cntVertexShaderFiles,
        const char **tessControlShaderFiles,
        const SIZE_T cntTessControlShaderFiles,
        const char **tessEvalShaderFiles,
        const SIZE_T cntTessEvalShaderFiles,
        const char **geometryShaderFiles,
        const SIZE_T cntGeometryShaderFiles,
        const char **fragmentShaderFiles,
        const SIZE_T cntFragmentShaderFiles,
        bool insertLineDirective) {

    // using arrays for automatic cleanup when a 'read' throws an exception
    Array<StringA> vertexShaderSrcs(cntVertexShaderFiles);
    Array<StringA> tessControlShaderSrcs(cntTessControlShaderFiles);
    Array<StringA> tessEvalShaderSrcs(cntTessEvalShaderFiles);
    Array<StringA> geometryShaderSrcs(cntGeometryShaderFiles);
    Array<StringA> fragmentShaderSrcs(cntFragmentShaderFiles);

    for(SIZE_T i = 0; i < cntVertexShaderFiles; i++) {
        if (!vislib::sys::ReadTextFile(vertexShaderSrcs[i], 
                vertexShaderFiles[i])) {
            return false;
        }
    }

    for(SIZE_T i = 0; i < cntTessControlShaderFiles; i++) {
        if (!vislib::sys::ReadTextFile(tessControlShaderSrcs[i], 
                tessControlShaderFiles[i])) {
            return false;
        }
    }

    for(SIZE_T i = 0; i < cntTessEvalShaderFiles; i++) {
        if (!vislib::sys::ReadTextFile(tessEvalShaderSrcs[i], 
                tessEvalShaderFiles[i])) {
            return false;
        }
    }

    for(SIZE_T i = 0; i < cntGeometryShaderFiles; i++) {
        if (!vislib::sys::ReadTextFile(geometryShaderSrcs[i], 
                geometryShaderFiles[i])) {
            return false;
        }
    }

    for(SIZE_T i = 0; i < cntFragmentShaderFiles; i++) {
        if (!vislib::sys::ReadTextFile(fragmentShaderSrcs[i], 
                fragmentShaderFiles[i])) {
            return false;
        }
    }

    // build up pointer arrays for attributes
    const char **vertexShaderSrcPtrs = new const char*[cntVertexShaderFiles];
    const char **tessControlShaderSrcPtrs
        = new const char*[cntTessControlShaderFiles];
    const char **tessEvalShaderSrcPtrs
        = new const char*[cntTessEvalShaderFiles];
    const char **geometryShaderSrcPtrs 
        = new const char*[cntGeometryShaderFiles];
    const char **fragmentShaderSrcPtrs 
        = new const char*[cntFragmentShaderFiles];

    try {
        for(SIZE_T i = 0; i < cntVertexShaderFiles; i++) {
            vertexShaderSrcPtrs[i] = vertexShaderSrcs[i].PeekBuffer();
        }
        for(SIZE_T i = 0; i < cntTessControlShaderFiles; i++) {
            tessControlShaderSrcPtrs[i] = tessControlShaderSrcs[i].PeekBuffer();
        }
        for(SIZE_T i = 0; i < cntTessEvalShaderFiles; i++) {
            tessEvalShaderSrcPtrs[i] = tessEvalShaderSrcs[i].PeekBuffer();
        }
        for(SIZE_T i = 0; i < cntGeometryShaderFiles; i++) {
            geometryShaderSrcPtrs[i] = geometryShaderSrcs[i].PeekBuffer();
        }
        for(SIZE_T i = 0; i < cntFragmentShaderFiles; i++) {
            fragmentShaderSrcPtrs[i] = fragmentShaderSrcs[i].PeekBuffer();
        }

        bool retval = this->Compile(vertexShaderSrcPtrs, cntVertexShaderFiles,
            tessControlShaderSrcPtrs, cntTessControlShaderFiles,
            tessEvalShaderSrcPtrs, cntTessEvalShaderFiles,
            geometryShaderSrcPtrs, cntGeometryShaderFiles, 
            fragmentShaderSrcPtrs, cntFragmentShaderFiles, 
            insertLineDirective);

        ARY_SAFE_DELETE(vertexShaderSrcPtrs);
        ARY_SAFE_DELETE(tessControlShaderSrcPtrs);
        ARY_SAFE_DELETE(tessEvalShaderSrcPtrs);
        ARY_SAFE_DELETE(geometryShaderSrcPtrs);
        ARY_SAFE_DELETE(fragmentShaderSrcPtrs);

        return retval;

        // free pointer arrays on exception
    } catch(OpenGLException e) { // catch OpenGLException to avoid truncating
        ARY_SAFE_DELETE(vertexShaderSrcPtrs);
        ARY_SAFE_DELETE(tessControlShaderSrcPtrs);
        ARY_SAFE_DELETE(tessEvalShaderSrcPtrs);
        ARY_SAFE_DELETE(geometryShaderSrcPtrs);
        ARY_SAFE_DELETE(fragmentShaderSrcPtrs);
        throw e;
    } catch(CompileException e) {
        ARY_SAFE_DELETE(vertexShaderSrcPtrs);
        ARY_SAFE_DELETE(tessControlShaderSrcPtrs);
        ARY_SAFE_DELETE(tessEvalShaderSrcPtrs);
        ARY_SAFE_DELETE(geometryShaderSrcPtrs);
        ARY_SAFE_DELETE(fragmentShaderSrcPtrs);
        throw e;
    } catch(Exception e) {
        ARY_SAFE_DELETE(tessControlShaderSrcPtrs);
        ARY_SAFE_DELETE(tessEvalShaderSrcPtrs);
        ARY_SAFE_DELETE(vertexShaderSrcPtrs);
        ARY_SAFE_DELETE(geometryShaderSrcPtrs);
        ARY_SAFE_DELETE(fragmentShaderSrcPtrs);
        throw e;
    } catch(...) {
        ARY_SAFE_DELETE(vertexShaderSrcPtrs);
        ARY_SAFE_DELETE(tessControlShaderSrcPtrs);
        ARY_SAFE_DELETE(tessEvalShaderSrcPtrs);
        ARY_SAFE_DELETE(geometryShaderSrcPtrs);
        ARY_SAFE_DELETE(fragmentShaderSrcPtrs);
        throw Exception("Unknown Exception", __FILE__, __LINE__);
    }

    return false; // should be unreachable code!
}


/*
 * vislib::graphics::gl::GLSLTesselationShader::SetProgramParameter
 */
void vislib::graphics::gl::GLSLTesselationShader::SetProgramParameter(GLenum name,
        GLint value) {
    ASSERT(GLSLShader::IsValidHandle(this->hProgObj));

    glProgramParameteriEXT(this->hProgObj, name, value);
}
