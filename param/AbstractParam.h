/*
 * AbstractParam.h
 *
 * Copyright (C) 2008 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_ABSTRACTPARAM_H_INCLUDED
#define MEGAMOLCORE_ABSTRACTPARAM_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "api/MegaMolCore.std.h"
#include "vislib/RawStorage.h"
#include "vislib/String.h"
#include "vislib/tchar.h"


namespace megamol {
namespace core {
namespace param {


    /** forward declaration of owning class */
    class AbstractParamSlot;


    /**
     * Abstract base class for all parameter objects
     */
    class MEGAMOLCORE_API AbstractParam {
    public:
        friend class AbstractParamSlot;

        /**
         * Dtor.
         */
        virtual ~AbstractParam(void);

        /**
         * Returns a machine-readable definition of the parameter.
         *
         * @param outDef A memory block to receive a machine-readable
         *               definition of the parameter.
         */
        virtual void Definition(vislib::RawStorage& outDef) const = 0;

        /**
         * Gets whether the parameter is visible in the gui.
         *
         * @return 'true' if the parameter is visible in the gui.
         */
        inline bool IsVisible(void) const {
            return this->visible;
        }

        /**
         * Tries to parse the given string as value for this parameter and
         * sets the new value if successful. This also triggers the update
         * mechanism of the slot this parameter is assigned to.
         *
         * @param v The new value for the parameter as string.
         *
         * @return 'true' on success, 'false' otherwise.
         */
        virtual bool ParseValue(const vislib::TString& v) = 0;

        /**
         * Returns the value of the parameter as string.
         *
         * @return The value of the parameter as string.
         */
        virtual vislib::TString ValueString(void) const = 0;

    protected:

        /**
         * Ctor.
         *
         * @param visible If 'true' the parameter is visible in the gui.
         */
        AbstractParam(bool visible = true);

        /**
         * Answers whether this parameter object is assigned to a public slot.
         *
         * @return 'true' if this parameter object is assigned to a public
         *         slot, 'false' otherwise.
         */
        bool isSlotPublic(void) const;

        /**
         * Sets the dirty flag of the owning parameter slot and might call the
         * update callback.
         */
        void setDirty(void);

    private:

        /** The holding slot */
        class AbstractParamSlot *slot;

        /** flag whether or not this parameter is visible from the gui */
        bool visible;

    };


} /* end namespace param */
} /* end namespace core */
} /* end namespace megamol */

#endif /* MEGAMOLCORE_ABSTRACTPARAM_H_INCLUDED */
