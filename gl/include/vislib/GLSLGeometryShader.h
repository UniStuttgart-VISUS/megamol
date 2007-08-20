/*
 * GLSLGeometryShader.h
 *
 * Copyright (C) 2006 - 2007 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#ifndef VISLIB_GLSLGEOMETRYSHADER_H_INCLUDED
#define VISLIB_GLSLGEOMETRYSHADER_H_INCLUDED
#if (_MSC_VER > 1000)
#pragma once
#endif /* (_MSC_VER > 1000) */
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
         * Compiles a new shader program object using a vertex shader and a
         * fragment shader. Both shader sources will be compiled into shader
         * objects and both will be attached to a program object. The program
         * object will not be linked. You must call 'Link' before you can use
         * the shader. Using 'vertexShaderSrc' as source code of the 
         * vertex shader and 'fragmentShaderSrc' as source code of the pixel
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
         * Compiles a new shader program object using a vertex shader and a
         * fragment shader. Both shader sources will be compiled into shader
         * objects and both will be attached to a program object. The program
         * object will not be linked. You must call 'Link' before you can use
         * the shader. Using the concatenation of the null terminated
         * strings in 'vertexShaderSrc' as source code of the vertex shader 
         * and the content of 'fragmentShaderSrc' as source code of the pixel
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

    protected:

    private:

    };
    
} /* end namespace gl */
} /* end namespace graphics */
} /* end namespace vislib */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_GLSLGEOMETRYSHADER_H_INCLUDED */

