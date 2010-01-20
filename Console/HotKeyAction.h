/*
 * HotKeyAction.h
 *
 * Copyright (C) 2009 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCON_HOTKEYACTION_H_INCLUDED
#define MEGAMOLCON_HOTKEYACTION_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

namespace megamol {
namespace console {

    /**
     * Base class for hotkey actions
     */
    class HotKeyAction {
    public:

        /**
         * Ctor
         */
        HotKeyAction(void);

        /**
         * Dtor
         */
        virtual ~HotKeyAction(void);

        /**
         * Aaaaaaaand Action!
         */
        virtual void Trigger(void) = 0;

    };

} /* end namespace console */
} /* end namespace megamol */

#endif /* MEGAMOLCON_HOTKEYACTION_H_INCLUDED */
