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

        typedef std::vector<std::string> FilePathExtensions_t;
        typedef uint FilePathFlags_t;
        enum FilePathFlags_ : uint32_t {
            Flag_File                 = 1 << 0,
            Flag_Directory            = 1 << 1,
            Flag_NoExistenceCheck     = 1 << 2,
            Flag_NoChange             = 1 << 3,
            Flag_RestrictedExtensions = 1 << 4,
            Flag_ToBeCreated          = Flag_NoExistenceCheck | Flag_NoChange
        };

        /**
         * Ctor.
         *
         * @param initVal The initial value
         * @param flags The flags for the parameter
         * @param exts The required file extensions for the parameter
         */
        FilePathParam(const std::string& initVal, FilePathFlags_t flags = Flag_File, FilePathExtensions_t exts = {});

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
        bool ParseValue(const vislib::TString& v) override;

        /**
         * Sets the value of the parameter and optionally sets the dirty flag
         * of the owning parameter slot.
         *
         * @param v the new value for the parameter
         * @param setDirty If 'true' the dirty flag of the owning parameter
         *                 slot is set and the update callback might be called.
         */
        void SetValue(const std::string& v, bool setDirty = true);
        void SetValue(const vislib::TString& v, bool setDirty = true);

        /**
         * Gets the value of the parameter
         *
         * @return The value of the parameter
         */
        inline vislib::TString Value() const {
            return vislib::TString(this->value.generic_u8string().c_str());
        }

        /**
         * Returns the value of the parameter as string.
         *
         * @return The value of the parameter as string.
         */
        vislib::TString ValueString() const override {
            return vislib::TString(this->value.generic_u8string().c_str());
        }

        /**
         * Gets the file path parameter flags
         *
         * @return The flags
         */
        inline FilePathFlags_t GetFlags() const {
            return this->flags;
        }

        /**
         * Gets the required file extensions
         *
         * @return The file extensions
         */
        inline const FilePathExtensions_t& GetExtensions() const {
            return this->extensions;
        }

    private:

        /** The flags of the parameter */
        FilePathFlags_t flags;

        /** The accepted file extension(s).
         * Leave empty to allow all extensions.
         * Use with Flag_RestrictedExtensions flag.
         */
        FilePathExtensions_t extensions;

        /** The file or directory path */
        std::filesystem::path value;

        /** Function checks if setting new file path value is valid depending on given flags */
        bool valid_change(std::string v);
    };


} /* end namespace param */
} /* end namespace core */
} /* end namespace megamol */

#endif /* MEGAMOLCORE_FILEPATHPARAM_H_INCLUDED */
