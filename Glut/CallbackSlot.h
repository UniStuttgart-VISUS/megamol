/*
 * CallbackSlot.h
 *
 * Copyright (C) 2008 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLVIEWER_CALLBACKSLOT_H_INCLUDED
#define MEGAMOLVIEWER_CALLBACKSLOT_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */


#include "MegaMolViewer.h"
#include "ApiHandle.h"
#include "vislib/Array.h"


namespace megamol {
namespace viewer {


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
        void Call(megamol::viewer::ApiHandle& caller, void *parameters = NULL);

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


} /* end namespace viewer */
} /* end namespace megamol */

#endif /* MEGAMOLVIEWER_CALLBACKSLOT_H_INCLUDED */
