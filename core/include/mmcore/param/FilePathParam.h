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
#include <filesystem>


namespace megamol {
namespace core {
namespace param {


    /**
     * class for file path parameter objects
     */
    class MEGAMOLCORE_API FilePathParam : public AbstractParam {
    public:

        /**  UTF-8 GUIDELINE:
        * - Convert your string object with std::filesystem::u8path() before passing to CTOR or SetValue()
        * - Convert std::filesystem::path received via Value() with .generic_u8string() if string is required or use .generic_u8string().c_str() if char pointer is required
        * - Pass std::filesystem::path to fopen with .native()
        */

        enum FilePathFlags_ : uint32_t {
            Flag_File                           = 1 << 0,
            Flag_Directory                      = 1 << 1,
            Flag_NoExistenceCheck               = 1 << 2,
            Flag_RestrictExtension              = 1 << 3,
            /// Convenience flags:
            Flag_File_RestrictExtension         = Flag_File | Flag_RestrictExtension,
            Flag_File_ToBeCreated               = Flag_File | Flag_NoExistenceCheck,
            Flag_File_ToBeCreatedWithRestrExts  = Flag_File | Flag_NoExistenceCheck | Flag_RestrictExtension,
            Flag_Directory_ToBeCreated          = Flag_Directory | Flag_NoExistenceCheck
        };

        typedef uint32_t Flags_t;

        typedef std::vector<std::string> Extensions_t;

        typedef std::function<void(const std::string&, std::weak_ptr<bool>, const std::string&)> RegisterNotificationCallback_t;

        /**
         * Ctor.
         *
         * @param initVal The initial value
         * @param flags The flags for the parameter
         * @param exts The required file extensions for the parameter
         */
        FilePathParam(const std::filesystem::path& initVal, Flags_t flags = Flag_File, const Extensions_t& exts = {});
        FilePathParam(const std::string& initVal, Flags_t flags = Flag_File, const Extensions_t& exts = {});
        FilePathParam(const char* initVal, Flags_t flags = Flag_File, const Extensions_t& exts = {});

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
        void SetValue(const std::filesystem::path& v, bool setDirty = true);
        void SetValue(const std::string& v, bool setDirty = true);
        void SetValue(const char* v, bool setDirty = true);

        /**
         * Gets the value of the parameter utf8 encoded for loading of files.
         *
         * @return The value of the parameter
         */
        std::filesystem::path Value() const {
            return this->value;
        }

        /**
         * Returns the value of the parameter as utf8 decoded string for storing in project file.
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

        /**
         * Register static notifications.
         */
        bool RegisterNotifications(const RegisterNotificationCallback_t& pc);
        // Check if registration is required
        inline bool RegisterNotifications() const {
            return !this->registered_notifications;
        }

        /**
         * Function checks if path is valid for given flags
         *
         * @return Return 0 for success, flags with failed check otherwise.
         */
        static Flags_t ValidatePath(const std::filesystem::path& p, const Extensions_t& , Flags_t f);

    private:

        /** The flags of the parameter */
        const Flags_t flags;

        /** The accepted file extension(s).
         * Leave empty to allow all extensions.
         * Only considered when Flag_RestrictExtension is set.
         */
        const Extensions_t extensions;

        /** The file or directory path */
        std::filesystem::path value;

        /** Indicates whether notifications are already registered or not (should be done only once) */
        bool registered_notifications;
        /** Flags for opening specific notifications */
        std::map<Flags_t, std::shared_ptr<bool>> open_notification;
    };


} /* end namespace param */
} /* end namespace core */
} /* end namespace megamol */

#endif /* MEGAMOLCORE_FILEPATHPARAM_H_INCLUDED */
