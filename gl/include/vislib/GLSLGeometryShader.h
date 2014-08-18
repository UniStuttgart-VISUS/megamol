/*
 * GLSLGeometryShader.h
 *
 * Copyright (C) 2006 - 2007 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#ifndef VISLIB_GLSLGEOMETRYSHADER_H_INCLUDED
#define VISLIB_GLSLGEOMETRYSHADER_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */


#include "vislib/GLSLShader.h"
#include "vislib/ExtensionsDependent.h"


namespace vislib {
namespace graphics {
namespace gl {


    /**
     * Class of GLSL geometry shaders.
     *
     * Use a 'Compile' Method to compile the shader sources, and use 'Link' to
     * construct a shader programm. After this was successful you can activate 
     * the shader using 'Enable'. Although you can access the shader programm 
     * handle directly it is recommended to use the classes methods where 
     * possible (e.g.: better use 'SetParameter' then 'glUniform').
     */
    class GLSLGeometryShader : public GLSLShader {
    public:

        /**
         * Answer the extensions that are required for ARB shaders as
         * space-separated ANSI strings.
         *
         * @return The extensions that are requiered for ARB shaders.
         */
        static const char * RequiredExtensions(void);

        /**
         * Initialise the extensions that are required for objects of the 
         * dependent class. This method must be called before using any objects
         * of the dependent classes.
         *
         * @return true, if all required extension could be loaded, 
         *         false otherwise.
         */
        static bool InitialiseExtensions(void) {
            return ExtensionsDependent<GLSLGeometryShader>
                ::InitialiseExtensions();
        }
    
        /**
         * Answers whether the required extensions are available.
         *
         * @return True if all required extensions are available and 
         *         initialisation should be successful, false otherwise.
         */
        static bool AreExtensionsAvailable(void) {
            return ExtensionsDependent<GLSLGeometryShader>
                ::AreExtensionsAvailable();
        }

        /** Shader code snippet enabling the gpu4 shader extension */
        static const char *GPU4_EXTENSION_DIRECTIVE;

        /** Ctor. */
        GLSLGeometryShader(void);

        /** Dtor. */
        virtual ~GLSLGeometryShader(void);

        /**
         * Compiles a new shader program object using a vertex shader, a 
         * geometry shader, and a fragment shader. All shader sources will be 
         * compiled into shader objects and they will be attached to a program
         * object. The program object will not be linked. You must call 'Link'
         * before you can use the shader. Using 'vertexShaderSrc' as source 
         * code of the vertex shader, 'geometryShaderSrc' as source code of the
         * geometry shader and 'fragmentShaderSrc' as source code of the pixel
         * shader.
         *
         * @param vertexShaderSrc   The null terminated source string of the 
         *                          vertex shader.
         * @param geometryShaderSrc The null terminated source string of the
         *                          geometry shader.
         * @param fragmentShaderSrc The null terminated source string of the
         *                          pixel shader.
         *
         * @return true if the shader was successfully compiled.
         *
         * @throws OpenGLException If an OpenGL call for compiling the shader
         *                         fails.
         */
        virtual bool Compile(const char *vertexShaderSrc, 
            const char *geometryShaderSrc, const char *fragmentShaderSrc);

        /**
         * Compiles a new shader program object using a vertex shader, a 
         * geometry shader, and a fragment shader. All shader sources will be 
         * compiled into shader objects and they will be attached to a program
         * object. The program object will not be linked. You must call 'Link'
         * before you can use the shader. Using 'vertexShaderSrc' as source 
         * code of the vertex shader, 'geometryShaderSrc' as source code of the
         * geometry shader and 'fragmentShaderSrc' as source code of the pixel
         * shader.
         *
         * @param vertexShaderSrc      An array of 'cntVertexShaderSrc' null
         *                             terminated strings forming the vertex
         *                             shader.
         * @param cntVertexShaderSrc   The number of elements in 
         *                             'vertexShaderSrc'.
         * @param geometryShaderSrc    An array of 'cntGeometryShaderSrc' null
         *                             terminated strings forming the geometry
         *                             shader.
         * @param cntGeometryShaderSrc The number of elements in
         *                             'geometryShaderSrc'.
         * @param fragmentShaderSrc    An array of 'cntFragmentShaderSrc' null
         *                             terminated strings forming the pixel
         *                             shader.
         * @param cntFragmentShaderSrc The number of elements in
         *                             'fragmentShaderSrc'.
         * @param insertLineDirective  Indicates whether the '#line' directive
         *                             should be inserted between each two
         *                             shader source strings.
         *
         * @return true if the shader was successfully compiled.
         *
         * @throws OpenGLException If an OpenGL call for compiling the shader
         *                         fails.
         */
        virtual bool Compile(const char **vertexShaderSrc, 
            const SIZE_T cntVertexShaderSrc, const char **geometryShaderSrc,
            const SIZE_T cntGeometryShaderSrc, const char **fragmentShaderSrc,
            const SIZE_T cntFragmentShaderSrc, 
            bool insertLineDirective = true);

        /**
         * Compiles a new shader program object using a vertex shader, a 
         * geometry shader, and a fragment shader. All shader sources will be 
         * compiled into shader objects and they will be attached to a program
         * object. The program object will not be linked. You must call 'Link'
         * before you can use the shader. Using 'vertexShaderFile' content as
         * source code of the vertex shader, 'geometryShaderFile' content as
         * source code of the geometry shader and 'fragmentShaderFile' content
         * as source code of the pixel shader.
         *
         * @param vertexShaderFile   The name of the vertex shader source file.
         * @param geometryShaderFile The name of the geometry shader source 
         *                           file.
         * @param fragmentShaderFile The name of the pixel shader source file.
         * 
         * @return true if the shader was successfully compiled.
         *
         * @throws OpenGLException If an OpenGL call for compiling the shader
         *                         fails.
         * @throws IOException     If reading the shader code from an open
         *                         file failed.
         */
        virtual bool CompileFromFile(const char *vertexShaderFile, 
            const char *geometryShaderFile, const char *fragmentShaderFile);

        /**
         * Compiles a new shader program object using a vertex shader, a 
         * geometry shader, and a fragment shader. All shader sources will be 
         * compiled into shader objects and they will be attached to a program
         * object. The program object will not be linked. You must call 'Link'
         * before you can use the shader. Using 'vertexShaderFiles' content as
         * source code of the vertex shader, 'geometryShaderFiles' content as
         * source code of the geometry shader and 'fragmentShaderFiles' content
         * as source code of the pixel shader.
         *
         * @param vertexShaderFiles      Array of names of the vertex shader 
         *                               source files.
         * @param cntVertexShaderFiles   Number of vertex shader source files
         * @param geometryShaderFiles    Array of names of the geometry shader 
         *                               source files.
         * @param cntGeometryShaderFiles Number of geometry shader source files
         * @param fragmentShaderFiles    Array of names of the fragment shader
         *                               source files.
         * @param cntFragmentShaderFiles Number of fragment shader source files
         * @param insertLineDirective    Indicates whether the '#line' 
         *                               directive should be inserted between 
         *                               each two shader source strings.
         * 
         * @return true if the shader was successfully compiled.
         *
         * @throws OpenGLException If an OpenGL call for compiling the shader
         *                         fails.
         * @throws IOException     If reading the shader code from an open
         *                         file failed.
         */
        virtual bool CompileFromFile(const char **vertexShaderFiles,
            const SIZE_T cntVertexShaderFiles,
            const char **geometryShaderFiles,
            const SIZE_T cntGeometryShaderFiles,
            const char **fragmentShaderFiles,
            const SIZE_T cntFragmentShaderFiles,
            bool insertLineDirective = true);

        /**
         * Sets a shader program parameter using 'glProgramParameteriEXT'. 
         * These parameters must be set before linking the shader. In general
         * the parameters 'GL_GEOMETRY_INPUT_TYPE_EXT', 
         * 'GL_GEOMETRY_OUTPUT_TYPE_EXT', and 'GL_GEOMETRY_VERTICES_OUT_EXT' 
         * must be set to appropriate values.
         *
         * @param name  The name constant of the parameter to be set.
         * @param value The value to set the parameter to.
         */
        void SetProgramParameter(GLenum name, GLint value);

    };
    
} /* end namespace gl */
} /* end namespace graphics */
} /* end namespace vislib */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_GLSLGEOMETRYSHADER_H_INCLUDED */
