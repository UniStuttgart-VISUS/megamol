/*
 * Instance.h
 *
 * Copyright (C) 2006 - 2011 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOL_WGL_INSTANCE_H_INCLUDED
#define MEGAMOL_WGL_INSTANCE_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "ApiHandle.h"


namespace megamol {
namespace wgl {

    /** 
     * Library instance
     */
    class Instance : public ApiHandle {
    public:

        /** Ctor */
        Instance(void);

        /** Dtor */
        virtual ~Instance(void);

    private:

    };


} /* end namespace wgl */
} /* end namespace megamol */

#endif /* MEGAMOL_WGL_INSTANCE_H_INCLUDED */
