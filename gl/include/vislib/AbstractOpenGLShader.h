/*
 * AbstractOpenGLShader.h
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */

#ifndef VISLIB_ABSTRACTOPENGLSHADER_H_INCLUDED
#define VISLIB_ABSTRACTOPENGLSHADER_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */


#include "glh/glh_genext.h"
#include "vislib/Exception.h"
#include "vislib/String.h"


namespace vislib {
namespace graphics {
namespace gl {


    /**
     * This class defines the interface for OpenGL shader classes.
     */
    class AbstractOpenGLShader {

    public:

        /**
         * This exception is throws, if compiling a shader failed.
         */
        class CompileException : public Exception {

        public:

            /** possible values for failed compile action */
            enum CompileAction {
                ACTION_UNKNOWN,
                ACTION_COMPILE_UNKNOWN,
                ACTION_COMPILE_VERTEX_CODE,
                ACTION_COMPILE_FRAGMENT_CODE,
                ACTION_COMPILE_GEOMETRY_CODE,
				ACTION_COMPILE_COMPUTE_CODE,
                ACTION_LINK
            };

            /**
             * Answer the correct 'CompileAction' value of a compilation failed
             * exception of a shader object of the type 'type'.
             *
             * @param type Specifies the type of shader object.
             *
             * @return The requested 'CompileAction' value.
             */
            static CompileAction CompilationFailedAction(GLenum type);

            /**
             * Answers a human readable name string for the given 
             * 'CompileAction' value.
             *
             * @param action The 'CompileAction' value.
             *
             * @return A human readable name string.
             */
            static const char* CompileActionName(CompileAction action);

            /**
             * Ctor.
             *
             * @param file The file the exception was thrown in.
             * @param line The line the exception was thrown in.
             */
            CompileException(const char *file, const int line);

            /**
             * Ctor.
             *
             * @param msg  The exception detail message.
             * @param file The file the exception was thrown in.
             * @param line The line the exception was thrown in.
             */
            CompileException(const char *msg, const char *file, 
                const int line);

            /**
             * Ctor.
             *
             * @param msg  The exception detail message.
             * @param file The file the exception was thrown in.
             * @param line The line the exception was thrown in.
             */
            CompileException(const wchar_t *msg, const char *file, 
                const int line);

            /**
             * Ctor.
             *
             * @param msg  The exception detail message.
             * @param action The failed compile action.
             * @param file The file the exception was thrown in.
             * @param line The line the exception was thrown in.
             */
            CompileException(const char *msg, CompileAction action, 
                const char *file, const int line);

            /**
             * Ctor.
             *
             * @param msg  The exception detail message.
             * @param action The failed compile action.
             * @param file The file the exception was thrown in.
             * @param line The line the exception was thrown in.
             */
            CompileException(const wchar_t *msg, CompileAction action, 
                const char *file, const int line);

            /**
             * Create a clone of 'rhs'.
             *
             * @param rhs The object to be cloned.
             */
            CompileException(const CompileException& rhs);

            /** Dtor. */
            virtual ~CompileException(void);

            /**
             * Answer the failed compile action.
             *
             * @return The failed compile action.
             */
            inline CompileAction FailedAction(void) {
                return this->action;
            }

            /**
             * Assignment operator.
             *
             * @param rhs The right hand side operand.
             *
             * @return *this.
             */
            virtual CompileException& operator =(const CompileException& rhs);

        private:

            /** The failed compile action */
            CompileAction action;

        }; /* end class CompileException */

        /** Dtor. */
        virtual ~AbstractOpenGLShader(void);

        //virtual GLenum Create(const char *src) = 0;

        //virtual GLenum CreateFromFile(const char *filename);

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
        virtual GLenum Disable(void) = 0;
        
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
        virtual GLenum Enable(void) = 0;

        /**
         * Releases all resources allocated by the shader. It is safe to
         * 
         * @return GL_NO_ERROR if the resource have been released or have
         *         never been allocated, an error code, if they have been
         *         allocated but could not be released.
         */
        virtual GLenum Release(void) = 0;

    protected:

        /** Disallow instances of this class. */
        AbstractOpenGLShader(void);

    };
    
} /* end namespace gl */
} /* end namespace graphics */
} /* end namespace vislib */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_OPENGLSHADER_H_INCLUDED */
