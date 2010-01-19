/*
 * IntParam.h
 *
 * Copyright (C) 2008 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_INTPARAM_H_INCLUDED
#define MEGAMOLCORE_INTPARAM_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "api/MegaMolCore.std.h"
#include "AbstractParam.h"


namespace megamol {
namespace core {
namespace param {


    /**
     * class for int parameter objects
     */
    class MEGAMOLCORE_API IntParam : public AbstractParam {
    public:

        /**
         * Ctor.
         *
         * @param initVal The initial value
         * @param visible If 'true' the parameter is visible in the gui.
         */
        IntParam(int initVal, bool visible = true);

        /**
         * Ctor.
         *
         * @param initVal The initial value
         * @param minVal The minimum value
         * @param visible If 'true' the parameter is visible in the gui.
         */
        IntParam(int initVal, int minVal, bool visible = true);

        /**
         * Ctor.
         *
         * @param initVal The initial value
         * @param minVal The minimum value
         * @param maxVal The maximum value
         * @param visible If 'true' the parameter is visible in the gui.
         */
        IntParam(int initVal, int minVal, int maxVal, bool visible = true);

        /**
         * Dtor.
         */
        virtual ~IntParam(void);

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
        void SetValue(int v, bool setDirty = true);

        /**
         * Gets the value of the parameter
         *
         * @return The value of the parameter
         */
        inline int Value(void) const {
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
        inline operator int(void) const {
            return this->val;
        }

    private:

        /** The value of the parameter */
        int val;

        /** The minimum value for the parameter */
        int minVal;

        /** The maximum value for the parameter */
        int maxVal;

    };


} /* end namespace param */
} /* end namespace core */
} /* end namespace megamol */

#endif /* MEGAMOLCORE_INTPARAM_H_INCLUDED */
