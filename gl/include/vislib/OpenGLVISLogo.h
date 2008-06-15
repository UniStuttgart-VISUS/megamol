/*
 * OpenGLVISLogo.h
 *
 * Copyright (C) 2006 - 2008 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#ifndef VISLIB_OPENGLVISLOGO_H_INCLUDED
#define VISLIB_OPENGLVISLOGO_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */


#include "vislib/AbstractVISLogo.h"


namespace vislib {
namespace graphics {
namespace gl {

    /**
     * Helper for rendering a VIS logo in OpenGL.
     */
    class OpenGLVISLogo : public AbstractVISLogo {

    public:

        /** Ctor. */
        OpenGLVISLogo(void);

        /** Dtor. */
        virtual ~OpenGLVISLogo(void);

        /**
         * Create all required resources for rendering a VIS logo.
         */
        virtual void Create(void);

        /**
         * Render the VIS logo. Create() must have been called before.
         */
        virtual void Draw(void);

        /**
         * Release all resources of the VIS logo.
         */
        virtual void Release(void);

    };

} /* end namespace gl */
} /* end namespace graphics */
} /* end namespace vislib */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_OPENGLVISLOGO_H_INCLUDED */
