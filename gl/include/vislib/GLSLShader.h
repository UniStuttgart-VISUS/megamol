/*
 * GLSLShader.h
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */

#ifndef VISLIB_GLSLSHADER_H_INCLUDED
#define VISLIB_GLSLSHADER_H_INCLUDED
#if (_MSC_VER > 1000)
#pragma once
#endif /* (_MSC_VER > 1000) */


#include "vislib/AbstractOpenGLShader.h"
#include "vislib/types.h"

#if (_MSC_VER > 1000)
#pragma warning(disable: 4996)
#endif /* (_MSC_VER > 1000) */
#include "glh/glh_extensions.h"
#if (_MSC_VER > 1000)
#pragma warning(default: 4996)
#endif /* (_MSC_VER > 1000) */


namespace vislib {
namespace graphics {
namespace gl {


    /**
     * TODO: comment class
     */
    class GLSLShader : public AbstractOpenGLShader {

    public:

        /**
         * Initialise the extensions that are required for GLSL shader objects. 
         * This method must be called before creating the first shader.
         *
         * @return true, if all required extension could be loaded, 
         *         false otherwise.
         */
        static bool InitialiseExtensions(void);

        /** Vertex shader source for fixed function transformation. */
        static const char *FTRANSFORM_VERTEX_SHADER_SRC;

        /** Ctor. */
        GLSLShader(void);

        /** Dtor. */
        ~GLSLShader(void);

        /**
         * Create a new shader using 'vertexShaderSrc' as source code of the 
         * vertex shader and 'fragmentShaderSrc' as source code of the pixel
         * shader.
         *
         * @param vertexShaderSrc   The null terminated source string of the 
         *                          vertex shader.
         * @param fragmentShaderSrc The null terminated source string of the
         *                          pixel shader.
         *
         * @return true, if the shader was successfully created.
         *
         * @throws OpenGLException If an OpenGL call for creating the shader
         *                         fails.
         */
        virtual bool Create(const char *vertexShaderSrc, 
            const char *fragmentShaderSrc);

        /**
         * Create a new shader using the concatenation of the null terminated
         * strings in 'vertexShaderSrc' as source code of the vertex shader 
         * and the content of 'fragmentShaderSrc' as source code of the pixel
         * shader.
         *
         * @param vertexShaderSrc      An array of 'cntVertexShaderSrc' null
         *                             terminated strings forming the vertex
         *                             shader.
         * @param cntVertexShaderSrc   The number of elements in 
         *                             'vertexShaderSrc'.
         * @param fragmentShaderSrc    An array of 'cntFragmentShaderSrc' null
         *                             terminated strings forming the pixel
         *                             shader.
         * @param cntFragmentShaderSrc The number of elements in
         *                             'fragmentShaderSrc'.
         *
         * @return true, if the shader was successfully created.
         *
         * @throws OpenGLException If an OpenGL call for creating the shader
         *                         fails.
         */
        virtual bool Create(const char **vertexShaderSrc, 
            const SIZE_T cntVertexShaderSrc, const char **fragmentShaderSrc,
            const SIZE_T cntFragmentShaderSrc);

        /**
         * Crate a new shader loading the shader code from two files.
         *
         * @param vertexShaderFile   The name of the vertex shader source file.
         * @param fragmentShaderFile The name of the pixel shader source file.
         *
         * @return true, if the shader was successfully created, false, if one
         *         of the shader files could not be opened.
         * 
         * @throws OpenGLException If an OpenGL call for creating the shader
         *                         fails.
         * @throws IOException     If reading the shader code from an open
         *                         file failed.
         */
        virtual bool CreateFromFile(const char *vertexShaderFile,
            const char *fragmentShaderFile);

        //virtual bool CreateFromFile(const char **vertexShaderFile,
        //    const SIZE_T cntVertexShaderFile, const char **fragmentShaderFile,
        //    const SIZE_T cntFragmentShaderFile);

        /**
         * Disables GLSL shaders.
         */
        virtual void Disable(void);
        
        /**
         * Enables the shader.
         *
         * @throws OpenGLException       If enabling the shader failed.
         * @throws IllegalStateException If this->type is TYPE_UNKNOWN.
         */
        virtual void Enable(void);

        /**
         * Releases all resources allocated by the shader.
         */
        virtual void Release(void);

    protected:

        /**
         * Answer the shader error string for the specified program object.
         *
         * @param hProg A handle to a program object.
         *
         * @return A string holding the compiler error string.
         *
         * @throws OpenGLException If the compile errors could not be retrieved, 
         *                         i. e. because 'hProg' is not a valid shader
         *                         program.
         */
        StringA getProgramInfoLog(GLhandleARB hProg);

        /** 
         * Answer the compile status of the program designated by 'hProg'.
         *
         * @param hProg A handle to a program object.
         *
         * @return true, if the program was successfully compiled, false 
         *         otherwise.
         *
         * @throws OpenGLException If the compile status could not be retrieved, 
         *                         i. e. because 'hProg' is not a valid shader
         *                         handle.
         */
        bool isCompiled(GLhandleARB hProg);

        /** 
         * Answer the linker status of the program designated by 'hProg'.
         *
         * @param hProg A handle to a program object.
         *
         * @return true, if the program was successfully linked, false 
         *         otherwise.
         *
         * @throws OpenGLException If the linker status could not be retrieved, 
         *                         i. e. because 'hProg' is not a valid shader
         *                         handle.
         */
        bool isLinked(GLhandleARB hProg);

        /** Handle of the program object. */
        GLhandleARB hProgObj;
    };
    
} /* end namespace gl */
} /* end namespace graphics */
} /* end namespace vislib */

#endif /* VISLIB_GLSLSHADER_H_INCLUDED */

