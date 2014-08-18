/*
 * GLSLComputeShader.h
 *
 * Copyright (C) 2006 - 2012 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#ifndef VISLIB_GLSLCOMPUTESHADER_H_INCLUDED
#define VISLIB_GLSLCOMPUTESHADER_H_INCLUDED
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
     * Class of GLSL compute shaders.
     *
     * Use a 'Compile' Method to compile the shader sources, and use 'Link' to
     * construct a shader programm. After this was successful you can activate 
     * the shader using 'Enable'. Although you can access the shader programm 
     * handle directly it is recommended to use the classes methods where 
     * possible (e.g.: better use 'SetParameter' then 'glUniform').
     */
    class GLSLComputeShader : public GLSLShader {
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
            return ExtensionsDependent<GLSLComputeShader>
                ::InitialiseExtensions();
        }
    
        /**
         * Answers whether the required extensions are available.
         *
         * @return True if all required extensions are available and 
         *         initialisation should be successful, false otherwise.
         */
        static bool AreExtensionsAvailable(void) {
            return ExtensionsDependent<GLSLComputeShader>
                ::AreExtensionsAvailable();
        }

        /** Shader code snippet enabling the gpu4 shader extension */
        //static const char *GPU4_EXTENSION_DIRECTIVE;

        /** Ctor. */
        GLSLComputeShader(void);

        /** Dtor. */
        virtual ~GLSLComputeShader(void);

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
         * @param computeShaderSrc  The null terminated source string of the 
         *                          compute shader.
         *
         * @return true if the shader was successfully compiled.
         *
         * @throws OpenGLException If an OpenGL call for compiling the shader
         *                         fails.
         */
        virtual bool Compile(const char *computeShaderSrc);

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
         * @param computeShaderSrc     An array of 'cntComputeShaderSrc' null
         *                             terminated strings forming the vertex
         *                             shader.
         * @param cntComputeShaderSrc  The number of elements in 
         *                             'computeShaderSrc'.
         * @param insertLineDirective  Indicates whether the '#line' directive
         *                             should be inserted between each two
         *                             shader source strings.
         *
         * @return true if the shader was successfully compiled.
         *
         * @throws OpenGLException If an OpenGL call for compiling the shader
         *                         fails.
         */
        virtual bool Compile(const char **computeShaderSrc, 
            const SIZE_T cntComputeShaderSrc, 
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
         * @param computeShaderFile   The name of the compute shader source file.
         * @return true if the shader was successfully compiled.
         *
         * @throws OpenGLException If an OpenGL call for compiling the shader
         *                         fails.
         * @throws IOException     If reading the shader code from an open
         *                         file failed.
         */
        virtual bool CompileFromFile(const char *computeShaderFile);

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
         * @param computeShaderFiles     Array of names of the compute shader 
         *                               source files.
         * @param cntComputeShaderFiles  Number of compute shader source files
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
        virtual bool CompileFromFile(const char **computeShaderFiles,
            const SIZE_T cntComputeShaderFiles,
            bool insertLineDirective = true);


		/**
		 * Starts the compute shader with the number of Groups specified.
		 * The compute shader program object must be compiled, linked and 
		 * enabled be dispatched correctly.
		 * See OpenGL specification for details. 
		 *
		 * @param numGroupsX number of thread groups in X direction
		 * @param numGroupsY number of thread groups in Y direction
		 * @param numGroupsZ number of thread groups in Z direction
		 */
		virtual void Dispatch(unsigned int numGroupsX, unsigned int numGroupsY, unsigned int numGroupsZ);
    };
    
} /* end namespace gl */
} /* end namespace graphics */
} /* end namespace vislib */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_GLSLCOMPUTESHADER_H_INCLUDED */
