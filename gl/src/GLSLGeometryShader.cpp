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
#include "vislib/glverify.h"


/*
 * vislib::graphics::gl::GLSLGeometryShader::RequiredExtensions
 */
const char * 
vislib::graphics::gl::GLSLGeometryShader::RequiredExtensions(void) {
    // would be better if it would call the super class
    return "GL_ARB_shader_objects GL_EXT_geometry_shader4 ";
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
        const char **vertexShaderSrc, const SIZE_T cntVertexShaderSrc, 
        const char **geometryShaderSrc, const SIZE_T cntGeometryShaderSrc, 
        const char **fragmentShaderSrc, const SIZE_T cntFragmentShaderSrc, 
        bool insertLineDirective) {

    USES_GL_VERIFY;
    ASSERT(vertexShaderSrc != NULL);
    ASSERT(fragmentShaderSrc != NULL);

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
