/*
 * ARBShader.h
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */

#ifndef VISLIB_ARBSHADER_H_INCLUDED
#define VISLIB_ARBSHADER_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */


#include "vislib/AbstractOpenGLShader.h"
#include "vislib/ExtensionsDependent.h"


namespace vislib {
namespace graphics {
namespace gl {


    /**
     * TODO: comment class
     */
    class ARBShader : public AbstractOpenGLShader, public ExtensionsDependent<ARBShader> {

    public:

        /** The possible types of an ARB shader. */
        enum Type {
            TYPE_UNKNOWN = 0,
            TYPE_VERTEX_SHADER = GL_VERTEX_PROGRAM_ARB,
            TYPE_FRAGMENT_SHADER = GL_FRAGMENT_PROGRAM_ARB
        };


        /**
         * Answer the extensions that are required for ARB shaders as
         * space-separated ANSI strings.
         *
         * @return The extensions that are requiered for ARB shaders.
         */
        static const char * RequiredExtensions(void);

        /** Ctor. */
        ARBShader(void);

        /** Dtor. */
        virtual ~ARBShader(void);

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
         *
         * @throws IllegalStateException If the shader is not valid, i. e. has
         *                               not been successfully created.
         */
        virtual GLenum Enable(void);

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
         * Releases all resources allocated by the shader. It is safe to
         * 
         * @return GL_NO_ERROR if the resource have been released or have
         *         never been allocated, an error code, if they have been
         *         allocated but could not be released.
         */
        virtual GLenum Release(void);

        /**
         * Set the four components of the local parameter 'name'.
         *
         * The shader must have been enabled before using this method.
         *
         * @param name The name of the parameter.
         * @param v1   The value of the first component (x/r).
         * @param v2   The value of the second component (y/g).
         * @param v3   The value of the third component (z/b).
         * @param v4   The value of the fourth component (w/a). 
         *
         * @return GL_NO_ERROR in case of success, an error code otherwise.
         */
        virtual GLenum SetParameter(const GLuint name, const double v1, 
            const double v2 = 0.0, const double v3 = 0.0, 
            const double v4 = 0.0);

        /**
         * Set the four components of the local parameter 'name'. The array
         * 'v' must be a non NULL vector of at least four elements.
         *
         * The shader must have been enabled before using this method.
         *
         * @param name The name of the parameter.
         * @param v    The array holding the values to pass to the shader.
         *
         * @return GL_NO_ERROR in case of success, an error code otherwise.
         */
        virtual GLenum SetParameter(const GLuint name, const double *v);

        /**
         * Set the four components of the local parameter 'name'.
         *
         * The shader must have been enabled before using this method.
         *
         * @param name The name of the parameter.
         * @param v1   The value of the first component (x/r).
         * @param v2   The value of the second component (y/g).
         * @param v3   The value of the third component (z/b).
         * @param v4   The value of the fourth component (w/a). 
         *
         * @return GL_NO_ERROR in case of success, an error code otherwise.
         */
        virtual GLenum SetParameter(const GLuint name, const float v1, 
            const float v2 = 0.0f, const float v3 = 0.0f, 
            const float v4 = 0.0f);

        /**
         * Set the four components of the local parameter 'name'. The array
         * 'v' must be a non NULL vector of at least four elements.
         *
         * The shader must have been enabled before using this method.
         *
         * @param name The name of the parameter.
         * @param v    The array holding the values to pass to the shader.
         *
         * @return GL_NO_ERROR in case of success, an error code otherwise.
         */
        virtual GLenum SetParameter(const GLuint name, const float *v) = 0;

        /**
         * Set the four components of the local parameter 'name'.
         *
         * The shader must have been enabled before using this method.
         *
         * @param name The name of the parameter.
         * @param v1   The value of the first component (x/r).
         * @param v2   The value of the second component (y/g).
         * @param v3   The value of the third component (z/b).
         * @param v4   The value of the fourth component (w/a). 
         *
         * @return GL_NO_ERROR in case of success, an error code otherwise.
         */
        inline GLenum SetParameter(const GLuint name, const int v1, 
                const int v2 = 0, const int v3 = 0, const int v4 = 0) {
            return this->SetParameter(name, static_cast<double>(v1), 
                static_cast<double>(v2), static_cast<double>(v3), 
                static_cast<double>(v4));
        }

        /**
         * Set the four components of the local parameter 'name'. The array
         * 'v' must be a non NULL vector of at least four elements.
         *
         * The shader must have been enabled before using this method.
         *
         * @param name The name of the parameter.
         * @param v    The array holding the values to pass to the shader.
         *
         * @return GL_NO_ERROR in case of success, an error code otherwise.
         */
        virtual GLenum SetParameter(const GLuint name, const int *v);

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

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_ARBSHADER_H_INCLUDED */
