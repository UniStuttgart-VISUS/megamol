/*
 * ButtonParam.h
 *
 * Copyright (C) 2008 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_BUTTONPARAM_H_INCLUDED
#define MEGAMOLCORE_BUTTONPARAM_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "api/MegaMolCore.std.h"
#include "AbstractParam.h"
#include "vislib/types.h"
#include "vislib/sys/KeyCode.h"


namespace megamol {
namespace core {
namespace param {


    /**
     * Special class for parameter objects representing a button. These
     * objects have no value at all, but trigger the update callback of the
     * slot whenever the button in the gui is pressed.
     */
    class MEGAMOLCORE_API ButtonParam : public AbstractParam {
    public:

        /**
         * Ctor.
         *
         * @param title The human readable title of the button.
         * @param key The prefered key for the button (if any). Be away, that
         *            if you do not assign a key, the button will not be
         *            available from viewers without a GUI.
         */
        ButtonParam(WORD key = 0);

        /**
         * Ctor.
         *
         * @param title The human readable title of the button.
         * @param key The prefered key for the button (if any). Be away, that
         *            if you do not assign a key, the button will not be
         *            available from viewers without a GUI.
         */
        ButtonParam(const vislib::sys::KeyCode &key);

        /**
         * Dtor.
         */
        virtual ~ButtonParam(void);

        /**
         * Returns a machine-readable definition of the parameter.
         *
         * @param outDef A memory block to receive a machine-readable
         *               definition of the parameter.
         */
        virtual void Definition(vislib::RawStorage& outDef) const;

        /**
         * Tries to parse the given string as value for this parameter and
         * sets the new value if successful. This also triggers the update
         * mechanism of the slot this parameter is assigned to.
         *
         * @param v The new value for the parameter as string.
         *
         * @return 'true' on success, 'false' otherwise.
         */
        virtual bool ParseValue(const vislib::TString& v);

        /**
         * Returns the value of the parameter as string.
         *
         * @return The value of the parameter as string.
         */
        virtual vislib::TString ValueString(void) const;

    private:

#ifdef _WIN32
#pragma warning (disable: 4251)
#endif /* _WIN32 */
        /** The key of this button */
        vislib::sys::KeyCode key;
#ifdef _WIN32
#pragma warning (default: 4251)
#endif /* _WIN32 */

    };


} /* end namespace param */
} /* end namespace core */
} /* end namespace megamol */

#endif /* MEGAMOLCORE_BUTTONPARAM_H_INCLUDED */
