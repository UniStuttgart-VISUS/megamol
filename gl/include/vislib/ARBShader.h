/*
 * ARBShader.h
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */

#ifndef VISLIB_ARBSHADER_H_INCLUDED
#define VISLIB_ARBSHADER_H_INCLUDED
#if (_MSC_VER > 1000)
#pragma once
#endif /* (_MSC_VER > 1000) */


#include "vislib/AbstractOpenGLShader.h"

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
    class ARBShader : public AbstractOpenGLShader {

    public:

        /** The possible types of an ARB shader. */
        enum Type {
            TYPE_UNKNOWN = 0,
            TYPE_VERTEX_SHADER = GL_VERTEX_PROGRAM_ARB,
            TYPE_FRAGMENT_SHADER = GL_FRAGMENT_PROGRAM_ARB
        };


        /**
         * Initialise the extensions that are required for ARB shaders. This 
         * method must be called before creating the first shader.
         *
         * @return true, if all required extension could be loaded, 
         *         false otherwise.
         */
        static bool InitialiseExtensions(void);

        /** Ctor. */
        ARBShader(void);

        /** Dtor. */
        ~ARBShader(void);

        /**
         *
         * @throws CompileException If the shader source 'src' could not be 
         *                          compiled.
         * @throws OpenGLException  If an OpenGL error occurred during 
         *                          construction of the shader.
         */
        virtual bool Create(const char *src);

        /**
         *
         * @throws CompileException If the shader source 'src' could not be 
         *                          compiled.
         * @throws OpenGLException  If an OpenGL error occurred during 
         *                          construction of the shader.
         * @throws IOException      If reading from the shader file failed.
         */
        virtual bool CreateFromFile(const char *filename);

        /**
         * Disable ARB shaders.
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
         * Answer the type of this shader. The return value is only valid after
         * a shader has been successfully created.
         *
         * @return The type of this shader.
         */
        inline Type GetType(void) const {
            return this->type;
        }

        /**
         * Releases all resources allocated by the shader.
         */
        virtual void Release(void);

    protected:

        /**
         * The source code of a fragment shader must begin with this string for
         * being recognised as fragment shader.
         */
        static const char *FRAGMENT_SHADER_TOKEN;

        /**
         * The source code of a vertex shader must begin with this string for 
         * being recognised as a vertex shader.
         */
        static const char *VERTEX_SHADER_TOKEN;

        /**
         * Forbidden copy ctor.
         *
         * @param rhs The object to be cloned.
         *
         * @throws UnsupportedOperationException Unconditionally.
         */
        ARBShader(const ARBShader& rhs);

        /**
         * Forbidden assignment.
         *
         * @param rhs The right hand side operand.
         *
         * @return *this
         *
         * @throws IllegalParamException If (this != &rhs).
         */
        ARBShader& operator =(const ARBShader& rhs);

        /** The ID of the shader. */
        GLuint id;

        /** The type of this shader. */
        Type type;

    };
    
} /* end namespace gl */
} /* end namespace graphics */
} /* end namespace vislib */

#endif /* VISLIB_ARBSHADER_H_INCLUDED */

