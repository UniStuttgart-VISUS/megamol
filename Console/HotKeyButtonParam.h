/*
 * HotKeyButtonParam.h
 *
 * Copyright (C) 2009 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCON_HOTKEYBUTTONPARAM_H_INCLUDED
#define MEGAMOLCON_HOTKEYBUTTONPARAM_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "CoreHandle.h"
#include "HotKeyAction.h"
#include "vislib/SmartPtr.h"


namespace megamol {
namespace console {

    /**
     * Base class for hotkey actions
     */
    class HotKeyButtonParam: public HotKeyAction {
    public:

        /**
         * Ctor
         */
        HotKeyButtonParam(void);

        /**
         * Ctor
         *
         * @param hParam The core handle to the parameter slot
         */
        HotKeyButtonParam(const vislib::SmartPtr<CoreHandle>& hParam);

        /**
         * Dtor
         */
        virtual ~HotKeyButtonParam(void);

        /**
         * Access the parameter handle
         *
         * @return A reference to the parameter handle
         */
        inline vislib::SmartPtr<CoreHandle>& HParam(void) {
            return this->hParam;
        }

        /**
         * Access the parameter handle
         *
         * @return A reference to the parameter handle
         */
        inline const vislib::SmartPtr<CoreHandle>& HParam(void) const {
            return this->hParam;
        }

        /**
         * Aaaaaaaand Action!
         */
        virtual void Trigger(void);

    private:

        /** The parameter handle */
        vislib::SmartPtr<CoreHandle> hParam;

    };

} /* end namespace console */
} /* end namespace megamol */

#endif /* MEGAMOLCON_HOTKEYBUTTONPARAM_H_INCLUDED */
