/*
 * GLSLGeometryShader.cpp
 *
 * Copyright (C) 2006 - 2007 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#ifdef _WIN32
#include <windows.h>
#endif /* _WIN32 */

#include "vislib/GLSLTesselationShader.h"

#include "vislib/Array.h"
#include "vislib/glverify.h"
#include "the/memory.h"
#include "the/string.h"
#include "vislib/sysfunctions.h"


/*
 * vislib::graphics::gl::GLSLTesselationShader::RequiredExtensions
 */
const char * 
vislib::graphics::gl::GLSLTesselationShader::RequiredExtensions(void) {
    static the::astring exts = the::astring(
        vislib::graphics::gl::GLSLShader::RequiredExtensions())
        + " GL_ARB_gpu_shader5 GL_ARB_tessellation_shader ";
    return exts.c_str();
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
            const char **vertexShaderSrc, const size_t cntVertexShaderSrc,
            const char **tessControlShaderSrc,
            const size_t cntTessControlShaderSrc,
            const char **tessEvalShaderSrc, const size_t cntTessEvalShaderSrc,
            const char **geometryShaderSrc, const size_t cntGeometryShaderSrc,
            const char **fragmentShaderSrc, const size_t cntFragmentShaderSrc,
            bool insertLineDirective) {

    USES_GL_VERIFY;
    THE_ASSERT(vertexShaderSrc != NULL);
    //THE_ASSERT(tessControlShaderSrc != NULL);
    //THE_ASSERT(tessEvalShaderSrc != NULL);
    //THE_ASSERT(geometryShaderSrc != NULL);
    THE_ASSERT(fragmentShaderSrc != NULL);

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
    GL_VERIFY_THROW(this->hProgObj = ::glCreateProgramObjectARB());
    GL_VERIFY_THROW(::glAttachObjectARB(this->hProgObj, hVertexShader));
    if (cntTessControlShaderSrc != 0) {
        GL_VERIFY_THROW(::glAttachObjectARB(this->hProgObj, hTessControlShader));
    }
    if (cntTessEvalShaderSrc != 0) {
        GL_VERIFY_THROW(::glAttachObjectARB(this->hProgObj, hTessEvalShader));
    }
    if (cntGeometryShaderSrc != 0) {
        GL_VERIFY_THROW(::glAttachObjectARB(this->hProgObj, hGeometryShader));
    }
    GL_VERIFY_THROW(::glAttachObjectARB(this->hProgObj, hPixelShader));

    return true;
}


/*
 * vislib::graphics::gl::GLSLTesselationShader::CompileFromFile
 */
bool vislib::graphics::gl::GLSLTesselationShader::CompileFromFile(
        const char *vertexShaderFile,
        const char *tessControlShaderFile, const char *tessEvalShaderFile,
        const char *geometryShaderFile, const char *fragmentShaderFile) {
    the::astring vertexShaderSrc;
    the::astring tessControlShaderSrc;
    the::astring tessEvalShaderSrc;
    the::astring geometryShaderSrc;
    the::astring fragmentShaderSrc;

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

    return this->Compile(vertexShaderSrc.c_str(),
        tessControlShaderFile == NULL ? NULL : tessControlShaderSrc.c_str(),
        tessEvalShaderFile == NULL ? NULL : tessEvalShaderSrc.c_str(),
        geometryShaderFile == NULL ? NULL : geometryShaderSrc.c_str(),
        fragmentShaderSrc.c_str());
}


/*
 * vislib::graphics::gl::GLSLTesselationShader::CompileFromFile
 */
bool vislib::graphics::gl::GLSLTesselationShader::CompileFromFile(
        const char **vertexShaderFiles,
        const size_t cntVertexShaderFiles,
        const char **tessControlShaderFiles,
        const size_t cntTessControlShaderFiles,
        const char **tessEvalShaderFiles,
        const size_t cntTessEvalShaderFiles,
        const char **geometryShaderFiles,
        const size_t cntGeometryShaderFiles,
        const char **fragmentShaderFiles,
        const size_t cntFragmentShaderFiles,
        bool insertLineDirective) {

    // using arrays for automatic cleanup when a 'read' throws an exception
    Array<the::astring> vertexShaderSrcs(cntVertexShaderFiles);
    Array<the::astring> tessControlShaderSrcs(cntTessControlShaderFiles);
    Array<the::astring> tessEvalShaderSrcs(cntTessEvalShaderFiles);
    Array<the::astring> geometryShaderSrcs(cntGeometryShaderFiles);
    Array<the::astring> fragmentShaderSrcs(cntFragmentShaderFiles);

    for(size_t i = 0; i < cntVertexShaderFiles; i++) {
        if (!vislib::sys::ReadTextFile(vertexShaderSrcs[i], 
                vertexShaderFiles[i])) {
            return false;
        }
    }

    for(size_t i = 0; i < cntTessControlShaderFiles; i++) {
        if (!vislib::sys::ReadTextFile(tessControlShaderSrcs[i], 
                tessControlShaderFiles[i])) {
            return false;
        }
    }

    for(size_t i = 0; i < cntTessEvalShaderFiles; i++) {
        if (!vislib::sys::ReadTextFile(tessEvalShaderSrcs[i], 
                tessEvalShaderFiles[i])) {
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
        for(size_t i = 0; i < cntVertexShaderFiles; i++) {
            vertexShaderSrcPtrs[i] = vertexShaderSrcs[i].c_str();
        }
        for(size_t i = 0; i < cntTessControlShaderFiles; i++) {
            tessControlShaderSrcPtrs[i] = tessControlShaderSrcs[i].c_str();
        }
        for(size_t i = 0; i < cntTessEvalShaderFiles; i++) {
            tessEvalShaderSrcPtrs[i] = tessEvalShaderSrcs[i].c_str();
        }
        for(size_t i = 0; i < cntGeometryShaderFiles; i++) {
            geometryShaderSrcPtrs[i] = geometryShaderSrcs[i].c_str();
        }
        for(size_t i = 0; i < cntFragmentShaderFiles; i++) {
            fragmentShaderSrcPtrs[i] = fragmentShaderSrcs[i].c_str();
        }

        bool retval = this->Compile(vertexShaderSrcPtrs, cntVertexShaderFiles,
            tessControlShaderSrcPtrs, cntTessControlShaderFiles,
            tessEvalShaderSrcPtrs, cntTessEvalShaderFiles,
            geometryShaderSrcPtrs, cntGeometryShaderFiles, 
            fragmentShaderSrcPtrs, cntFragmentShaderFiles, 
            insertLineDirective);

        the::safe_array_delete(vertexShaderSrcPtrs);
        the::safe_array_delete(tessControlShaderSrcPtrs);
        the::safe_array_delete(tessEvalShaderSrcPtrs);
        the::safe_array_delete(geometryShaderSrcPtrs);
        the::safe_array_delete(fragmentShaderSrcPtrs);

        return retval;

        // free pointer arrays on exception
    } catch(OpenGLException e) { // catch OpenGLException to avoid truncating
        the::safe_array_delete(vertexShaderSrcPtrs);
        the::safe_array_delete(tessControlShaderSrcPtrs);
        the::safe_array_delete(tessEvalShaderSrcPtrs);
        the::safe_array_delete(geometryShaderSrcPtrs);
        the::safe_array_delete(fragmentShaderSrcPtrs);
        throw e;
    } catch(CompileException e) {
        the::safe_array_delete(vertexShaderSrcPtrs);
        the::safe_array_delete(tessControlShaderSrcPtrs);
        the::safe_array_delete(tessEvalShaderSrcPtrs);
        the::safe_array_delete(geometryShaderSrcPtrs);
        the::safe_array_delete(fragmentShaderSrcPtrs);
        throw e;
    } catch(the::exception e) {
        the::safe_array_delete(tessControlShaderSrcPtrs);
        the::safe_array_delete(tessEvalShaderSrcPtrs);
        the::safe_array_delete(vertexShaderSrcPtrs);
        the::safe_array_delete(geometryShaderSrcPtrs);
        the::safe_array_delete(fragmentShaderSrcPtrs);
        throw e;
    } catch(...) {
        the::safe_array_delete(vertexShaderSrcPtrs);
        the::safe_array_delete(tessControlShaderSrcPtrs);
        the::safe_array_delete(tessEvalShaderSrcPtrs);
        the::safe_array_delete(geometryShaderSrcPtrs);
        the::safe_array_delete(fragmentShaderSrcPtrs);
        throw the::exception("Unknown Exception", __FILE__, __LINE__);
    }

    return false; // should be unreachable code!
}


/*
 * vislib::graphics::gl::GLSLTesselationShader::SetProgramParameter
 */
void vislib::graphics::gl::GLSLTesselationShader::SetProgramParameter(GLenum name,
        GLint value) {
    THE_ASSERT(GLSLShader::IsValidHandle(this->hProgObj));

    glProgramParameteriEXT(this->hProgObj, name, value);
}
