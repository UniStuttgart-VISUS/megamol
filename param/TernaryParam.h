/*
 * TernaryParam.h
 *
 * Copyright (C) 2008 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_TERNARYPARAM_H_INCLUDED
#define MEGAMOLCORE_TERNARYPARAM_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "api/MegaMolCore.std.h"
#include "AbstractParam.h"
#include "vislib/Ternary.h"


namespace megamol {
namespace core {
namespace param {


    /**
     * class for ternary parameter objects
     */
    class MEGAMOLCORE_API TernaryParam : public AbstractParam {
    public:

        /**
         * Ctor.
         *
         * @param initVal The initial state value
         * @param visible If 'true' the parameter is visible in the gui.
         */
        TernaryParam(const vislib::math::Ternary& initVal, bool visible = true);

        /**
         * Dtor.
         */
        virtual ~TernaryParam(void);

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
         * Sets the value of the parameter and optionally sets the dirty flag
         * of the owning parameter slot.
         *
         * @param v the new value for the parameter
         * @param setDirty If 'true' the dirty flag of the owning parameter
         *                 slot is set and the update callback might be called.
         */
        void SetValue(vislib::math::Ternary v, bool setDirty = true);

        /**
         * Gets the value of the parameter
         *
         * @return The value of the parameter
         */
        inline const vislib::math::Ternary& Value(void) const {
            return this->val;
        }

        /**
         * Returns the value of the parameter as string.
         *
         * @return The value of the parameter as string.
         */
        virtual vislib::TString ValueString(void) const;

        /**
         * Gets the value of the parameter
         *
         * @return The value of the parameter
         */
        inline operator vislib::math::Ternary(void) const {
            return this->val;
        }

    private:

#ifdef _WIN32
#pragma warning (disable: 4251)
#endif /* _WIN32 */
        /** The value of the parameter */
        vislib::math::Ternary val;
#ifdef _WIN32
#pragma warning (default: 4251)
#endif /* _WIN32 */

    };


} /* end namespace param */
} /* end namespace core */
} /* end namespace megamol */

#endif /* MEGAMOLCORE_TERNARYPARAM_H_INCLUDED */
