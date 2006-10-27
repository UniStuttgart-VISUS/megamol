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


namespace vislib {
namespace graphics {
namespace gl {


    /**
     * This class defines the interface for OpenGL shader classes.
     */
    class AbstractOpenGLShader {

    public:

        /** Dtor. */
        virtual ~AbstractOpenGLShader(void);

        virtual GLenum Create(const char *src) = 0;

        virtual GLenum CreateFromFile() = 0;

        virtual void Disable(void) = 0;
        
        virtual void Enable(void) = 0;



    protected:

        /** Disallow instances of this class. */
        AbstractOpenGLShader(void);

    };
    
} /* end namespace gl */
} /* end namespace graphics */
} /* end namespace vislib */

#endif /* VISLIB_OPENGLSHADER_H_INCLUDED */
