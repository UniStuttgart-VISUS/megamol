/**
 * MegaMol
 * Copyright (c) 2010, MegaMol Dev Team
 * All rights reserved.
 */

#pragma once

#include <filesystem>

#include "AbstractParam.h"

namespace megamol::core::param {

/// See MegaMol development guide for utf8 related explanations ///

/**
 * class for file path parameter objects
 */
class FilePathParam : public AbstractParam {
public:
    enum FilePathFlags_ : uint32_t {
        Flag_File = 1 << 0,                    // Only allows to hold an existing file
        Flag_Directory = 1 << 1,               // Only allows to hold an existing directory
        Internal_NoExistenceCheck = 1 << 2,    // Only used internally - do not use as flag
        Internal_RestrictExtension = 1 << 3,   // Only used internally - do not use as flag
        Flag_Any = Flag_File | Flag_Directory, // Only allows to hold an existing file or directory
        Flag_Any_ToBeCreated = Flag_Any | Internal_NoExistenceCheck, // Allows to hold a non-existing file or directory
        Flag_File_ToBeCreated = Flag_File | Internal_NoExistenceCheck, // Allows to hold a non-existing file
        Flag_Directory_ToBeCreated =                                   // Allows to hold a non-existing directory
        Flag_Directory | Internal_NoExistenceCheck,
        Flag_File_RestrictExtension = // Allows to hold an existing file having one of the given extensions
        Flag_File | Internal_RestrictExtension,
        Flag_File_ToBeCreatedWithRestrExts = // Allows to hold a non-existing file having one of the given extensions
        Flag_File | Internal_NoExistenceCheck | Internal_RestrictExtension
    };

    typedef uint32_t Flags_t;
    typedef std::vector<std::string> Extensions_t;

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
     * Tries to parse the given string as value for this parameter and
     * sets the new value if successful. This also triggers the update
     * mechanism of the slot this parameter is assigned to.
     *
     * @param v The new value for the parameter as string.
     *
     * @return 'true' on success, 'false' otherwise.
     */
    bool ParseValue(std::string const& v) override;

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
        return this->GetAbsolutePathValue(this->value);
    }

    /**
     * Returns the value of the parameter as utf8 decoded string for storing in project file.
     *
     * @return The value of the parameter as string.
     */
    std::string ValueString() const override {
        return this->Value().generic_u8string();
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
     * Function checks if path is valid for given flags
     *
     * @return Return 0 for success, flags with failed check otherwise.
     */
    static Flags_t ValidatePath(const std::filesystem::path& p, const Extensions_t&, Flags_t f);


    /**
     * Adds absolute path to current project directory to the file path parameter.
     * This way file paths relative to the project directory can be resolved.
     */
    void SetProjectDirectory(const std::filesystem::path& p);

    /**
     * Returns either the current path value if it is an absolute path,
     * or concatinates project directory path and current path value if it is a relative path.
     */
    std::filesystem::path GetAbsolutePathValue(const std::filesystem::path& p) const;

private:
    /** The flags of the parameter */
    Flags_t flags;

    /** The accepted file extension(s).
     * Leave empty to allow all extensions.
     * Only considered when Flag_RestrictExtension is set.
     */
    Extensions_t extensions;

    /** The file or directory path */
    std::filesystem::path value;

    /**
     * Absolute path to project directory. If empty, no project path available.
     * This path is relative per guarantee of the frontend
     */
    std::filesystem::path project_directory;
};


} // namespace megamol::core::param
