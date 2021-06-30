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

#include "mmcore/utility/FileUtils.h"
#include "mmcore/api/MegaMolCore.std.h"
#include "AbstractParam.h"


namespace megamol {
namespace core {
namespace param {


    /**
     * class for file path parameter objects
     */
    class MEGAMOLCORE_API FilePathParam : public AbstractParam {
    public:

        typedef uint FilePathFlags;
        enum FilePathFlags_ : uint32_t {
            Flag_File               = 1 << 0,
            Flag_Directory          = 1 << 1,
            Flag_NoExistenceCheck   = 1 << 2,
            Flag_NoChange           = 1 << 3,
            Flag_ToBeCreated        = Flag_NoExistenceCheck | Flag_NoChange
        };

        /**
         * Ctor.
         *
         * @param initVal The initial value
         * @param flags The flags for the parameter
         */
        explicit FilePathParam(const std::filesystem::path& initVal, FilePathFlags flags = Flag_File);
        explicit FilePathParam(const std::string& initVal, FilePathFlags flags = Flag_File);
        explicit FilePathParam(const std::wstring& initVal, FilePathFlags flags = Flag_File);

        /**
         * Dtor.
         */
        ~FilePathParam() override = default;

        /**
         * Returns a machine-readable definition of the parameter.
         *
         * @param outDef A memory block to receive a machine-readable
         *               definition of the parameter.
         */
        void Definition(vislib::RawStorage& outDef) const override;

        /**
         * Tries to parse the given string as value for this parameter and
         * sets the new value if successful. This also triggers the update
         * mechanism of the slot this parameter is assigned to.
         *
         * @param v The new value for the parameter as string.
         *
         * @return 'true' on success, 'false' otherwise.
         */
        bool ParseValue(const vislib::TString& v) final;

        /**
         * Sets the value of the parameter and optionally sets the dirty flag
         * of the owning parameter slot.
         *
         * @param v the new value for the parameter
         * @param setDirty If 'true' the dirty flag of the owning parameter
         *                 slot is set and the update callback might be called.
         */
        void SetValue(const std::filesystem::path& v, bool setDirty = true);
        void SetValue(const std::string& v, bool setDirty = true);
        void SetValue(const std::wstring& v, bool setDirty = true);

        /**
         * Gets the value of the parameter
         *
         * @return The value of the parameter
         */
        inline const std::filesystem::path& Value() const {
            return this->val;
        }

        /**
         * Returns the value of the parameter as string.
         *
         * @return The value of the parameter as string.
         */
        vislib::TString ValueString() const override;

        /**
         * Gets the value of the parameter
         *
         * @return The value of the parameter
         */
        explicit inline operator const std::filesystem::path&() const {
            return this->val;
        }

        /**
         * ...
         *
         * @return ...
         */
        inline FilePathFlags GetFlags() const {
            return this->flags;
        }

    private:

        /** The flags of the parameter */
        FilePathFlags flags;

        /** The file or directory path */
        std::filesystem::path val;
    };


} /* end namespace param */
} /* end namespace core */
} /* end namespace megamol */

#endif /* MEGAMOLCORE_FILEPATHPARAM_H_INCLUDED */
