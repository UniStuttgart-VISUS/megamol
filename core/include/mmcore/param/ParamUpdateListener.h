/*
 * ParamUpdateListener.h
 *
 * Copyright (C) 2010 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_PARAMUPDATELISTENER_H_INCLUDED
#define MEGAMOLCORE_PARAMUPDATELISTENER_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "mmcore/param/ParamSlot.h"


namespace megamol {
namespace core {
namespace param {


    /**
     * Abstract base class for all parameter objects
     */
    class MEGAMOLCORE_API ParamUpdateListener {
    public:

        /** Ctor */
        ParamUpdateListener(void);

        /** Dtor. */
        virtual ~ParamUpdateListener(void);

        /**
         * Callback called when a parameter is updated
         *
         * @param slot The parameter updated
         */
        virtual void ParamUpdated(ParamSlot& slot) = 0;

    };


} /* end namespace param */
} /* end namespace core */
} /* end namespace megamol */

#endif /* MEGAMOLCORE_PARAMUPDATELISTENER_H_INCLUDED */
