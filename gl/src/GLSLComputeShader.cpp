/*
 * GLSLComputeShader.cpp
 *
 * Copyright (C) 2006 - 2012 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#ifdef _WIN32
#include <windows.h>
#endif /* _WIN32 */

#include "vislib/GLSLComputeShader.h"

#include "vislib/Array.h"
#include "vislib/glverify.h"
#include "the/memory.h"
#include "the/string.h"
#include "vislib/sysfunctions.h"


/*
 * vislib::graphics::gl::GLSLComputeShader::RequiredExtensions
 */
const char * 
vislib::graphics::gl::GLSLComputeShader::RequiredExtensions(void) {
    static the::astring exts = the::astring(
        vislib::graphics::gl::GLSLShader::RequiredExtensions())
        + " GL_VERSION_4_3 ";
    return exts.c_str();
}


/*
 * vislib::graphics::gl::GLSLComputeShader::GPU4_EXTENSION_DIRECTIVE
 
const char *
vislib::graphics::gl::GLSLComputeShader::GPU4_EXTENSION_DIRECTIVE =
    "#extension GL_EXT_gpu_shader4:enable\n";
	*/

/*
 * vislib::graphics::gl::GLSLComputeShader::GLSLComputeShader
 */
vislib::graphics::gl::GLSLComputeShader::GLSLComputeShader(void) 
        : GLSLShader() {
}


/*
 * vislib::graphics::gl::GLSLComputeShader::~GLSLComputeShader
 */
vislib::graphics::gl::GLSLComputeShader::~GLSLComputeShader(void) {
}


/*
 * vislib::graphics::gl::GLSLComputeShader::Compile
 */
bool vislib::graphics::gl::GLSLComputeShader::Compile(
        const char *computeShaderSrc) {
    const char *c[] = { computeShaderSrc };
    
    return this->Compile(c, 1);
}


/*
 * vislib::graphics::gl::GLSLComputeShader::Compile
 */
bool vislib::graphics::gl::GLSLComputeShader::Compile(
        const char **computeShaderSrc, const size_t cntComputeShaderSrc, 
        bool insertLineDirective) {

    USES_GL_VERIFY;
    THE_ASSERT(computeShaderSrc != NULL);

    this->Release();

	GLhandleARB computeShader = this->compileNewShader(GL_COMPUTE_SHADER,
        computeShaderSrc, static_cast<GLsizei>(cntComputeShaderSrc), 
        insertLineDirective);

    /* Assemble program object. */
    GL_VERIFY_THROW(this->hProgObj = ::glCreateProgramObjectARB());
    GL_VERIFY_THROW(::glAttachObjectARB(this->hProgObj, computeShader));
    GL_VERIFY_THROW(glLinkProgram(this->hProgObj));

    return true;
}


/*
 * vislib::graphics::gl::GLSLComputeShader::CompileFromFile
 */
bool vislib::graphics::gl::GLSLComputeShader::CompileFromFile(
        const char *computeShaderFile) {
    the::astring computeShaderSrc;

    if (!vislib::sys::ReadTextFile(computeShaderSrc, computeShaderFile)) {
        return false;
    }

    return this->Compile(computeShaderSrc.c_str());
}


/*
 * vislib::graphics::gl::GLSLComputeShader::CompileFromFile
 */
bool vislib::graphics::gl::GLSLComputeShader::CompileFromFile(
        const char **computeShaderFiles, const size_t cntComputeShaderFiles,
        bool insertLineDirective) {

    // using arrays for automatic cleanup when a 'read' throws an exception
    Array<the::astring> copmuteShaderSrcs(cntComputeShaderFiles);

    for(size_t i = 0; i < cntComputeShaderFiles; i++) {
        if (!vislib::sys::ReadTextFile(copmuteShaderSrcs[i], 
                computeShaderFiles[i])) {
            return false;
        }
    }
	
    // built up pointer arrays for attributes
    const char **computeShaderSrcPtrs = new const char*[cntComputeShaderFiles];

    try {
        for(size_t i = 0; i < cntComputeShaderFiles; i++) {
            computeShaderSrcPtrs[i] = copmuteShaderSrcs[i].c_str();
        }

        bool retval = this->Compile(computeShaderSrcPtrs, cntComputeShaderFiles,
            insertLineDirective);

        the::safe_array_delete(computeShaderSrcPtrs);

        return retval;

        // free pointer arrays on exception
    } catch(OpenGLException e) { // catch OpenGLException to avoid truncating
        the::safe_array_delete(computeShaderSrcPtrs);
        throw e;
    } catch(CompileException e) {
        the::safe_array_delete(computeShaderSrcPtrs);
        throw e;
    } catch(the::exception e) {
        the::safe_array_delete(computeShaderSrcPtrs);
        throw e;
    } catch(...) {
        the::safe_array_delete(computeShaderSrcPtrs);
        throw the::exception("Unknown Exception", __FILE__, __LINE__);
    }

    return false; // should be unreachable code!
}


void vislib::graphics::gl::GLSLComputeShader::Dispatch(unsigned int numGroupsX, unsigned int numGroupsY, unsigned int numGroupsZ)
{
	glDispatchCompute(numGroupsX, numGroupsY, numGroupsZ);
}
