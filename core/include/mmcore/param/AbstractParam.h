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

#include "mmcore/api/MegaMolCore.std.h"
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

        /**
         * Must be public for Button Press - Manuel Graeber
         * Sets the dirty flag of the owning parameter slot and might call the
         * update callback.
         */
        void setDirty(void);

        /**
         * Returns the value of the hash.
         *
         * @return The value of the hash.
         */
        inline size_t GetHash(void) const {
            return this->hash;
        }

        /**
         * Sets the value of the hash.
         *
         * @param hash The value of the hash.
         */
        inline void SetHash(const size_t &hash) {
            this->hash = hash;
        }

        /**
        * Answer visibility in GUI.
        *
        * @return GUI visibility
        */
        inline bool IsGUIVisible() const {
            return this->gui_visible;
        }

        /**
        * Set visibility in GUI.
        *
        * @param visible True: visible in GUI, false: invisible
        */
        inline void SetGUIVisible(const bool visible) {
            this->gui_visible = visible;
        }

        /**
        * Answer accessibility in GUI.
        *
        * @return GUI accessibility
        */
        inline bool IsGUIReadOnly() const {
            return this->gui_read_only;
        }

        /**
        * Set accessibility in GUI.
        *
        * @param read_only True: read-only in GUI, false: writable
        */
        inline void SetGUIReadOnly(const bool read_only) {
            this->gui_read_only = read_only;
        }

        /**
         * Returns the has_changed flag and resets the flag to false.
         *
         * @return has_changed
         */
        bool ConsumeHasChanged() {
            auto val = has_changed;
            has_changed = false;
            return val;
        }

    protected:

        /**
         * Ctor.
         */
        AbstractParam(void);

        /**
         * Answers whether this parameter object is assigned to a public slot.
         *
         * @return 'true' if this parameter object is assigned to a public
         *         slot, 'false' otherwise.
         */
        bool isSlotPublic(void) const;

        /**
         * Set has_changed flag to true.
         */
        void indicateChange() { has_changed = true; }

    private:

        /** The holding slot */
        class AbstractParamSlot *slot;

        /**
         * Hash indicating fundamental changes in parameter definition
         * (i.e. requires rebuilding the UI).
         */
        size_t hash;

        /**
        * Show or hide the parameter in the GUI.
        */
        bool gui_visible;

        /**
        * Make parameter read-only in the GUI.
        */
        bool gui_read_only;

        /**
         * Indicating that the value has changed.
         */
        bool has_changed;

    };


} /* end namespace param */
} /* end namespace core */
} /* end namespace megamol */

#endif /* MEGAMOLCORE_ABSTRACTPARAM_H_INCLUDED */
