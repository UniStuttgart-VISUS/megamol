/**
 * MegaMol
 * Copyright (c) 2008, MegaMol Dev Team
 * All rights reserved.
 */

#include "mmcore/param/FilePathParam.h"

#include "mmcore/utility/FileUtils.h"

using namespace megamol::core::param;


FilePathParam::FilePathParam(const std::filesystem::path& initVal, Flags_t flags, const Extensions_t& exts)
        : AbstractParam()
        , flags(flags)
        , extensions(exts)
        , value() {

    this->InitPresentation(AbstractParamPresentation::ParamType::FILEPATH);
    this->SetValue(initVal);
}


FilePathParam::FilePathParam(const std::string& initVal, Flags_t flags, const Extensions_t& exts)
        : FilePathParam(std::filesystem::u8path(initVal), flags, exts){};


FilePathParam::FilePathParam(const char* initVal, Flags_t flags, const Extensions_t& exts)
        : FilePathParam(std::filesystem::u8path(initVal), flags, exts){};


bool FilePathParam::ParseValue(std::string const& v) {

    try {
        this->SetValue(v);
        return true;
    } catch (...) {}
    return false;
}


void FilePathParam::SetValue(const std::filesystem::path& v, bool setDirty) {

    try {
        auto tmp_val_str = v.generic_u8string();
        std::replace(tmp_val_str.begin(), tmp_val_str.end(), '\\', '/');
        auto new_value = std::filesystem::path(tmp_val_str);
        if (this->value != new_value) {
            auto absolute_new_value = GetAbsolutePathValue(new_value);
            auto error_flags = FilePathParam::ValidatePath(absolute_new_value, this->extensions, this->flags);

            if (error_flags & Flag_File) {
                megamol::core::utility::log::Log::DefaultLog.WriteWarn(
                    "[FilePathParam] Omitting value '%s'. Expected file but directory is given.",
                    new_value.generic_u8string().c_str());
            }
            if (error_flags & Flag_Directory) {
                megamol::core::utility::log::Log::DefaultLog.WriteWarn(
                    "[FilePathParam] Omitting value '%s'. Expected directory but file is given.",
                    new_value.generic_u8string().c_str());
            }
            if (error_flags & Internal_NoExistenceCheck) {
                megamol::core::utility::log::Log::DefaultLog.WriteWarn(
                    "[FilePathParam] Omitting value '%s'. File does not exist.", new_value.generic_u8string().c_str());
            }
            if (error_flags & Internal_RestrictExtension) {
                std::string log_exts;
                for (auto& ext : this->extensions) {
                    log_exts += "'." + ext + "' ";
                }
                megamol::core::utility::log::Log::DefaultLog.WriteWarn(
                    "[FilePathParam] Omitting value '%s'. File does not have required extension: %s",
                    new_value.generic_u8string().c_str(), log_exts.c_str());
            }
            if (error_flags == 0) {
                this->value = new_value;
                this->indicateParamChange();
                if (setDirty)
                    this->setDirty();
            }
        }
    } catch (std::filesystem::filesystem_error& e) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "Filesystem Error: %s [%s, %s, line %d]\n", e.what(), __FILE__, __FUNCTION__, __LINE__);
    }
}


void megamol::core::param::FilePathParam::SetValue(const std::string& v, bool setDirty) {

    this->SetValue(std::filesystem::u8path(v), setDirty);
}


void megamol::core::param::FilePathParam::SetValue(const char* v, bool setDirty) {

    this->SetValue(std::filesystem::u8path(v), setDirty);
}


FilePathParam::Flags_t FilePathParam::ValidatePath(const std::filesystem::path& p, const Extensions_t& e, Flags_t f) {

    try {
        FilePathParam::Flags_t retval = 0;
        if ((f & FilePathParam::Flag_Any) != FilePathParam::Flag_Any) {
            if ((f & FilePathParam::Flag_File) && std::filesystem::is_directory(p)) {
                retval |= FilePathParam::Flag_File;
            }
            if ((f & FilePathParam::Flag_Directory) && std::filesystem::is_regular_file(p)) {
                retval |= FilePathParam::Flag_Directory;
            }
        }
        if (!(f & Internal_NoExistenceCheck) && !std::filesystem::exists(p)) {
            retval |= FilePathParam::Internal_NoExistenceCheck;
        }
        if (f & FilePathParam::Internal_RestrictExtension) {
            bool valid_ext = false;
            for (auto& ext : e) {
                if (p.extension().generic_u8string() == std::string("." + ext)) {
                    valid_ext = true;
                }
            }
            if (!valid_ext) {
                retval |= FilePathParam::Internal_RestrictExtension;
            }
        }
        return retval;
    } catch (std::filesystem::filesystem_error& e) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "Filesystem Error: %s [%s, %s, line %d]\n", e.what(), __FILE__, __FUNCTION__, __LINE__);
        return 0;
    }
}

void FilePathParam::SetProjectDirectory(const std::filesystem::path& p) {
    assert(p.is_absolute());

    this->project_directory = p;
}

std::filesystem::path FilePathParam::GetAbsolutePathValue(const std::filesystem::path& p) const {
    if (p.empty() || p.is_absolute())
        return p;

    auto concat = this->project_directory / p;

    return concat;
}
