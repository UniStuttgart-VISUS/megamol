/*
 * Window.h
 *
 * Copyright (C) 2006 - 2011 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOL_WGL_WINDOW_H_INCLUDED
#define MEGAMOL_WGL_WINDOW_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "ApiHandle.h"
#include "Instance.h"


namespace megamol {
namespace wgl {

    /** 
     * OpenGL rendering window
     */
    class Window : public ApiHandle {
    public:

        /** Ctor */
        Window(Instance& inst);

        /** Dtor */
        virtual ~Window(void);

    private:

        /** The library instance */
        Instance& inst;

    };


} /* end namespace wgl */
} /* end namespace megamol */

#endif /* MEGAMOL_WGL_WINDOW_H_INCLUDED */
