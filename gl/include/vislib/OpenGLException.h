/*
 * OpenGLException.h
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */

#ifndef VISLIB_OPENGLEXCEPTION_H_INCLUDED
#define VISLIB_OPENGLEXCEPTION_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */


#include "glh/glh_genext.h"

#include "vislib/Exception.h"


namespace vislib {
namespace graphics {
namespace gl {


    /**
     * An exception class that represents an OpenGL error.
     */
    class OpenGLException : public Exception {

    public:

        /**
         * Ctor.
         *
		 * @param errorCode An OpenGL error code.
         * @param file      The file the exception was thrown in.
         * @param line      The line the exception was thrown in.
         */
        OpenGLException(const GLenum errorCode, const char *file,
            const int line);

        /**
         * Create an exception using the error code returned by ::glGetError().
         *
         * @param file The file the exception was thrown in.
         * @param line The line the exception was thrown in.
         */
        OpenGLException(const char *file, const int line);

        /**
         * Create a clone of 'rhs'.
         *
         * @param rhs The object to be cloned.
         */
        OpenGLException(const OpenGLException& rhs);

        /** Dtor. */
        virtual ~OpenGLException(void);

        /**
         * Answer the OpenGL error code.
         *
         * @return The error code.
         */
        inline GLenum GetErrorCode(void) const {
            return this->errorCode;
        }

        /**
         * Assignment operator.
         *
         * @param rhs The right hand side operand.
         *
         * @return *this.
         */
        OpenGLException& operator =(const OpenGLException& rhs);

    private:

        /** The OpenGL error code represented by this exception. */
        GLenum errorCode;

    };

} /* end namespace gl */
} /* end namespace graphics */
} /* end namespace vislib */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_OPENGLEXCEPTION_H_INCLUDED */
