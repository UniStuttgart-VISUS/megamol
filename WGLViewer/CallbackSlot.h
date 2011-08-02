/*
 * CallbackSlot.h
 *
 * Copyright (C) 2008-2011 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOL_WGL_CALLBACKSLOT_H_INCLUDED
#define MEGAMOL_WGL_CALLBACKSLOT_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */


#include "MegaMolViewer.h"
#include "ApiHandle.h"
#include "vislib/Array.h"


namespace megamol {
namespace wgl {


    /**
     * Class of callback slots
     */
    class CallbackSlot {
    public:

        /** Ctor. */
        CallbackSlot(void);

        /** Dtor */
        ~CallbackSlot(void);

        /**
         * Calls this slot.
         *
         * @param caller The caller.
         * @param params Pointer to the parameters.
         */
        void Call(ApiHandle& caller, void *parameters = NULL);

        /**
         * Clears the callback slot
         */
        void Clear(void);

        /**
         * Registers a function on this slot.
         *
         * @param function The function to be registered.
         */
        void Register(mmvCallback function);

        /**
         * Unregisters a function from this slot. It is safe to unregister a
         * function never registered.
         *
         * @param function The function to be unregistered.
         */
        void Unregister(mmvCallback function);

    private:

        /** The first function to be called. */
        mmvCallback one;

        /** The other functions to be called. */
        vislib::Array<mmvCallback> *other;

    };


} /* end namespace wgl */
} /* end namespace megamol */

#endif /* MEGAMOL_WGL_CALLBACKSLOT_H_INCLUDED */
