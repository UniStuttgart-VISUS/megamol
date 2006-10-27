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

namespace vislib {
namespace graphics {
namespace gl {


    /**
     * TODO: comment class
     */
    class ARBShader : public AbstractOpenGLShader {

    public:

        /** Ctor. */
        ARBShader(void);

        /** Dtor. */
        ~ARBShader(void);

    protected:

    private:

    };
    
} /* end namespace gl */
} /* end namespace graphics */
} /* end namespace vislib */

#endif /* VISLIB_ARBSHADER_H_INCLUDED */

