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
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */


#include "vislib/AbstractOpenGLShader.h"
#include "vislib/ExtensionsDependent.h"
#include "vislib/types.h"


namespace vislib {
namespace graphics {
namespace gl {


    /**
     * TODO: comment class
     */
    class GLSLShader : public AbstractOpenGLShader, public ExtensionsDependent<GLSLShader> {

    public:

        /**
         * Answer whether 'hProg' is a valid program object handle, that also 
         * has not been scheduled for deletion.
         * 
         * @param hProg A program handle.
         *
         * @return true, if 'hProg' is a valid handle, false otherwise.
         */
        static bool IsValidHandle(GLhandleARB hProg);

        /**
         * Answer the extensions that are required for ARB shaders as
         * space-separated ANSI strings.
         *
         * @return The extensions that are requiered for ARB shaders.
         */
        static const char * RequiredExtensions(void);

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
         * Disable the shader. This method changes the GL to render using
         * the fixed function pipeline.
         *
         * It is safe to call this method, if the shader has not been
         * successfully created or enable.
         *
         * @return GL_NO_ERROR in case of success, an error code, if the
         *         shader was active but could not be disabled.
         */
        virtual GLenum Disable(void);
        
        /**
         * Enables the shader. The shader must have been successfully created
         * before.
         *
         * @return GL_NO_ERROR in case of success, an error code, if the
         *         shader could not be enabled.
         */
        virtual GLenum Enable(void);

        /**
         * Releases all resources allocated by the shader. It is safe to
         * 
         * @return GL_NO_ERROR if the resource have been released or have
         *         never been allocated, an error code, if they have been
         *         allocated but could not be released.
         */
        virtual GLenum Release(void);

        virtual GLenum SetParameter(const char *name, const float v1);

        virtual GLenum SetParameter(const char *name, const float v1, 
            const float v2);

        virtual GLenum SetParameter(const char *name, const float v1, 
            const float v2, const float v3);

        virtual GLenum SetParameter(const char *name, const float v1,
            const float v2, const float v3, const float v4);

        virtual GLenum SetParameter(const char *name, const int v1);

        virtual GLenum SetParameter(const char *name, const int v1, 
            const int v2);

        virtual GLenum SetParameter(const char *name, const int v1, 
            const int v2, const int v3);

        virtual GLenum SetParameter(const char *name, const int v1,
            const int v2, const int v3, const int v4);

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

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_GLSLSHADER_H_INCLUDED */
