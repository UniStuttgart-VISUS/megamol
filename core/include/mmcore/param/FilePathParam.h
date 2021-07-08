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

        enum FilePathFlags_ : uint32_t {
            Flag_File                    = 1 << 0,
            Flag_Directory               = 1 << 1,
            Flag_NoExistenceCheck        = 1 << 2,
            Flag_NoChange                = 1 << 3,
            Flag_RestrictExtension       = 1 << 4,
            /// Convenience flags:
            Flag_File_RestrictExtension  = Flag_File | Flag_RestrictExtension,
            Flag_File_ToBeCreated        = Flag_File | Flag_NoExistenceCheck | Flag_NoChange,
            Flag_Directory_ToBeCreated   = Flag_Directory | Flag_NoExistenceCheck | Flag_NoChange
        };
        typedef std::function<void(std::string&, const std::string&)> PopUpCallback_t;
        typedef std::vector<std::string> Extensions_t;
        typedef uint32_t Flags_t;

        /**
         * Ctor.
         *
         * @param initVal The initial value
         * @param flags The flags for the parameter
         * @param exts The required file extensions for the parameter
         */
        FilePathParam(const std::string& initVal, Flags_t flags = Flag_File, Extensions_t exts = {});

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
        inline Flags_t GetFlags() const {
            return this->flags;
        }

        /**
         * Gets the required file extensions
         *
         * @return The file extensions
         */
        inline const Extensions_t& GetExtensions() const {
            return this->extensions;
        }

        void SetPopUpCallback(const PopUpCallback_t& pc) {
            this->popup_callback = pc;
        }

    private:

        /** The flags of the parameter */
        Flags_t flags;

        /** The accepted file extension(s).
         * Leave empty to allow all extensions.
         * Use with Flag_RestrictExtension flag.
         */
        Extensions_t extensions;

        /** The file or directory path */
        std::filesystem::path value;

        /** Function checks if setting new file path value is valid depending on given flags */
        bool valid_change(std::string v);

        /** Pop-up callback function propagating log messages to gui pop-up */
        PopUpCallback_t popup_callback;
    };


} /* end namespace param */
} /* end namespace core */
} /* end namespace megamol */

#endif /* MEGAMOLCORE_FILEPATHPARAM_H_INCLUDED */
