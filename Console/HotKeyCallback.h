/*
 * HotKeyCallback.h
 *
 * Copyright (C) 2009 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCON_HOTKEYCALLBACK_H_INCLUDED
#define MEGAMOLCON_HOTKEYCALLBACK_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "HotKeyAction.h"


namespace megamol {
namespace console {

    /**
     * Base class for hotkey actions
     */
    class HotKeyCallback: public HotKeyAction {
    public:

        /**
         * Ctor
         */
        HotKeyCallback(void (*callback)(void));

        /**
         * Dtor
         */
        virtual ~HotKeyCallback(void);

        /**
         * Aaaaaaaand Action!
         */
        virtual void Trigger(void);

    private:

        /** The callback to be called (back) */
        void (*callback)(void);

    };

} /* end namespace console */
} /* end namespace megamol */

#endif /* MEGAMOLCON_HOTKEYCALLBACK_H_INCLUDED */
