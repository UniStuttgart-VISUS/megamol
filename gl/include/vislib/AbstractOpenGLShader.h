/*
 * AbstractOpenGLShader.h
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */

#ifndef VISLIB_ABSTRACTOPENGLSHADER_H_INCLUDED
#define VISLIB_ABSTRACTOPENGLSHADER_H_INCLUDED
#if (_MSC_VER > 1000)
#pragma once
#endif /* (_MSC_VER > 1000) */


#include <GL/gl.h>

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
             * Create a clone of 'rhs'.
             *
             * @param rhs The object to be cloned.
             */
            CompileException(const CompileException& rhs);

            /** Dtor. */
            virtual ~CompileException(void);

            /**
             * Assignment operator.
             *
             * @param rhs The right hand side operand.
             *
             * @return *this.
             */
            virtual CompileException& operator =(const CompileException& rhs);

        }; /* end class CompileException */

        /** Dtor. */
        virtual ~AbstractOpenGLShader(void);

        //virtual GLenum Create(const char *src) = 0;

        //virtual GLenum CreateFromFile(const char *filename);

        /**
         * Disable the shader.
         *
         * @throws OpenGLException If disabling the shader failed.
         */
        virtual void Disable(void) = 0;
        
        /**
         * Enables the shader.
         *
         * @throws OpenGLException If enabling the shader failed.
         */
        virtual void Enable(void) = 0;

        /**
         * Releases all resources allocated by the shader.
         */
        virtual void Release(void) = 0;

    protected:

        /** Disallow instances of this class. */
        AbstractOpenGLShader(void);

        /**
         * Read the content of the file 'filename' into 'outSrc'. 'outSrc' is 
         * being erased by this operation.
         *
         * @param outStr   The string to receive the content.
         * @param filename The name of the file being read.
         *
         * @return true, if the file could be read, false, if the file was not 
         *         found or could not be opened.
         *
         * @throws IOException If reading from the file failed.
         */
        virtual bool read(StringA& outStr, const char *filename) const;

    };
    
} /* end namespace gl */
} /* end namespace graphics */
} /* end namespace vislib */

#endif /* VISLIB_OPENGLSHADER_H_INCLUDED */
