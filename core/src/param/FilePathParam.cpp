/*
 * FilePathParam.cpp
 *
 * Copyright (C) 2008 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
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


std::string FilePathParam::Definition() const {
    return "MMFILA";
}


bool FilePathParam::ParseValue(std::string const& v) {

    try {
        this->SetValue(v);
        return true;
    } catch (...) {}
    return false;
}


void FilePathParam::SetValue(const std::filesystem::path& v, bool setDirty) {

    const auto normalize_path = [](std::filesystem::path const& path) -> std::filesystem::path {
        auto tmp_val_str = path.generic_u8string();
        std::replace(tmp_val_str.begin(), tmp_val_str.end(), '\\', '/');
        return std::filesystem::path{tmp_val_str};
    };

    try {
        auto new_value = normalize_path(v);

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
                if (new_value.is_absolute() && !this->project_directory.empty()) {
                    // is new path in project directory?
                    // then represent as relative path
                    auto new_val_str = new_value.u8string();
                    auto proj_dir_str = project_directory.u8string();

                    auto val_it = new_value.begin();
                    auto proj_it = project_directory.begin();
                    while (*val_it == *proj_it) {
                        val_it++;
                        proj_it++;
                    }
                    if (*proj_it == *project_directory.end()) {
                        // new value is in project directory path
                        // collect the tail
                        std::filesystem::path project_relative_path;
                        while (val_it != new_value.end()) {
                            project_relative_path = project_relative_path / *val_it;
                            val_it++;
                        }

                        new_value = normalize_path(project_relative_path);
                    }
                }

                this->value = new_value;
                this->indicateChange();
                if (setDirty)
                    this->setDirty();
            } else {
                // send around initial value, because the GUI might get stuck on an invalid file name
                this->indicateChange();
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

    return this->project_directory / p;
}
