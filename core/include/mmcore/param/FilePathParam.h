/*
 * FilePathParam.h
 *
 * Copyright (C) 2010 by VISUS (Universitaet Stuttgart).
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_FILEPATHPARAM_H_INCLUDED
#define MEGAMOLCORE_FILEPATHPARAM_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "mmcore/api/MegaMolCore.std.h"
#include "AbstractParam.h"
#include "vislib/String.h"
#include "vislib/tchar.h"
#include "vislib/types.h"


namespace megamol {
namespace core {
namespace param {


    /**
     * class for file path parameter objects
     */
    class MEGAMOLCORE_API FilePathParam : public AbstractParam {
    public:

        /**
         * No flags set
         */
        static const UINT32 FLAG_NONE;

        /**
         * Flag that the file path must not be changed by the framework
         */
        static const UINT32 FLAG_NOPATHCHANGE;

        /**
         * Flag that the framework should not search for the file
         */
        static const UINT32 FLAG_NOEXISTANCECHECK;

        /**
         * Flags FLAG_NOPATHCHANGE and FLAG_NOEXISTANCECHECK combined
         */
        static const UINT32 FLAG_TOBECREATED;

        /**
         * Ctor.
         *
         * @param initVal The initial value
         * @param flags The flags for the parameter
         */
        FilePathParam(const vislib::StringA& initVal, UINT32 flags = FLAG_NONE);

        /**
         * Ctor.
         *
         * @param initVal The initial value
         * @param flags The flags for the parameter
         */
        FilePathParam(const vislib::StringW& initVal, UINT32 flags = FLAG_NONE);

        /**
         * Ctor.
         *
         * @param initVal The initial value
         * @param flags The flags for the parameter
         */
        FilePathParam(const char *initVal, UINT32 flags = FLAG_NONE);

        /**
         * Ctor.
         *
         * @param initVal The initial value
         * @param flags The flags for the parameter
         */
        FilePathParam(const wchar_t *initVal, UINT32 flags = FLAG_NONE);

        /**
         * Dtor.
         */
        virtual ~FilePathParam(void);

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
        void SetValue(const vislib::StringA& v, bool setDirty = true);

        /**
         * Sets the value of the parameter and optionally sets the dirty flag
         * of the owning parameter slot.
         *
         * @param v the new value for the parameter
         * @param setDirty If 'true' the dirty flag of the owning parameter
         *                 slot is set and the update callback might be called.
         */
        void SetValue(const vislib::StringW& v, bool setDirty = true);

        /**
         * Sets the value of the parameter and optionally sets the dirty flag
         * of the owning parameter slot.
         *
         * @param v the new value for the parameter
         * @param setDirty If 'true' the dirty flag of the owning parameter
         *                 slot is set and the update callback might be called.
         */
        void SetValue(const char *v, bool setDirty = true);

        /**
         * Sets the value of the parameter and optionally sets the dirty flag
         * of the owning parameter slot.
         *
         * @param v the new value for the parameter
         * @param setDirty If 'true' the dirty flag of the owning parameter
         *                 slot is set and the update callback might be called.
         */
        void SetValue(const wchar_t *v, bool setDirty = true);

        /**
         * Gets the value of the parameter
         *
         * @return The value of the parameter
         */
        inline const vislib::TString& Value(void) const {
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
        inline operator const vislib::TString&(void) const {
            return this->val;
        }

    private:

        /** The flags of the parameter */
        UINT32 flags;

#ifdef _WIN32
#pragma warning (disable: 4251)
#endif /* _WIN32 */
        /** The value of the parameter */
        vislib::TString val;
#ifdef _WIN32
#pragma warning (default: 4251)
#endif /* _WIN32 */

    };


} /* end namespace param */
} /* end namespace core */
} /* end namespace megamol */

#endif /* MEGAMOLCORE_FILEPATHPARAM_H_INCLUDED */
