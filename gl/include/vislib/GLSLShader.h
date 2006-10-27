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


#include "vislib/AbstractOpenGLShader.h"

namespace vislib {
namespace graphics {
namespace gl {


    /**
     * TODO: comment class
     */
    class GLSLShader : public AbstractOpenGLShader {

    public:

        /** Ctor. */
        GLSLShader(void);

        /** Dtor. */
        ~GLSLShader(void);

    protected:

    private:

    };
    
} /* end namespace gl */
} /* end namespace graphics */
} /* end namespace vislib */

#endif /* VISLIB_GLSLSHADER_H_INCLUDED */

