/*
 * FilePathParam.cpp
 *
 * Copyright (C) 2008 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "mmcore/param/FilePathParam.h"
#include "mmcore/utility/FileUtils.h"


using namespace megamol::core::param;


FilePathParam::FilePathParam(const std::filesystem::path& initVal, Flags_t flags, const Extensions_t& exts)
    : AbstractParam()
    , flags(flags)
    , extensions(exts)
    , value()
    , registered_notifications(false)
    , open_notification() {
    this->open_notification[Flag_File] = std::make_shared<bool>(false);
    this->open_notification[Flag_Directory] = std::make_shared<bool>(false);
    this->open_notification[Flag_NoExistenceCheck] = std::make_shared<bool>(false);
    this->open_notification[Flag_RestrictExtension] = std::make_shared<bool>(false);
    this->InitPresentation(AbstractParamPresentation::ParamType::FILEPATH);
    this->SetValue(initVal);
}


FilePathParam::FilePathParam(const std::string& initVal, Flags_t flags, const Extensions_t& exts)
        : FilePathParam(std::filesystem::u8path(initVal), flags, exts){
};


FilePathParam::FilePathParam(const char * initVal, Flags_t flags, const Extensions_t& exts)
        : FilePathParam(std::filesystem::u8path(initVal), flags, exts){
};


void FilePathParam::Definition(vislib::RawStorage& outDef) const {

    outDef.AssertSize(6);
#if defined(UNICODE) || defined(_UNICODE)
    memcpy(outDef.AsAt<char>(0), "MMFILW", 6);
#else  /* defined(UNICODE) || defined(_UNICODE) */
    memcpy(outDef.AsAt<char>(0), "MMFILA", 6);
#endif /* defined(UNICODE) || defined(_UNICODE) */
}


bool FilePathParam::ParseValue(const vislib::TString& v) {

    try {
        this->SetValue(std::string(v.PeekBuffer()));
        return true;
    } catch(...) {
    }
    return false;
}


void FilePathParam::SetValue(const std::filesystem::path& v, bool setDirty) {

    try {
        auto new_value = v;
        if (this->value != new_value) {
            auto error_flags = FilePathParam::ValidatePath(new_value, this->extensions, this->flags);

            if (error_flags & Flag_File) {
                megamol::core::utility::log::Log::DefaultLog.WriteWarn(
                    "[FilePathParam] Omitting value '%s'. Expected file but directory is given.", new_value.generic_u8string().c_str());
                if (this->open_notification[Flag_File] != nullptr)
                    *this->open_notification[Flag_File] = true;
            }
            if (error_flags & Flag_Directory) {
                megamol::core::utility::log::Log::DefaultLog.WriteWarn(
                    "[FilePathParam] Omitting value '%s'. Expected directory but file is given.", new_value.generic_u8string().c_str());
                if (this->open_notification[Flag_Directory] != nullptr)
                    *this->open_notification[Flag_Directory] = true;
            }
            if (error_flags & Flag_NoExistenceCheck) {
                megamol::core::utility::log::Log::DefaultLog.WriteWarn(
                    "[FilePathParam] Omitting value '%s'. File does not exist.", new_value.generic_u8string().c_str());
                if (this->open_notification[Flag_NoExistenceCheck] != nullptr)
                    *this->open_notification[Flag_NoExistenceCheck] = true;
            }
            if (error_flags & Flag_RestrictExtension) {
                std::string log_exts;
                for (auto& ext : this->extensions) {
                    log_exts += "'." + ext + "' ";
                }
                megamol::core::utility::log::Log::DefaultLog.WriteWarn(
                    "[FilePathParam] Omitting value '%s'. File does not have required extension: %s", new_value.generic_u8string().c_str(), log_exts.c_str());
                if (this->open_notification[Flag_RestrictExtension] != nullptr)
                    *this->open_notification[Flag_RestrictExtension] = true;
            }
            if (error_flags == 0) {
                this->value = new_value;
                this->indicateChange();
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
        if ((f & FilePathParam::Flag_File) && std::filesystem::is_directory(p)) { 
            retval |= FilePathParam::Flag_File;
        }
        if ((f & FilePathParam::Flag_Directory) && std::filesystem::is_regular_file(p)) {
            retval |= FilePathParam::Flag_Directory;
        }
        if (!(f & Flag_NoExistenceCheck) && !std::filesystem::exists(p)) {
            retval |= FilePathParam::Flag_NoExistenceCheck;
        }
        if (f & FilePathParam::Flag_RestrictExtension) {
            bool valid_ext = false;
            for (auto& ext : e) {
                if (p.extension().generic_u8string() == std::string("." + ext)) {
                    valid_ext = true;
                }
            }
            if (!valid_ext) {
                retval |= FilePathParam::Flag_RestrictExtension;
            }
        }
        return retval;
    }
    catch (std::filesystem::filesystem_error& e) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "Filesystem Error: %s [%s, %s, line %d]\n", e.what(), __FILE__, __FUNCTION__, __LINE__);
        return 0;
    }
}


bool FilePathParam::RegisterNotifications(const FilePathParam::RegisterNotificationCallback_t& pc) {

    if (!this->registered_notifications) {
        const std::string prefix = "Omitting value. ";
        pc("file_is_dir", std::weak_ptr<bool>(this->open_notification[Flag_File]), prefix + "Expected file but directory is given.");
        pc("dir_is_file", std::weak_ptr<bool>(this->open_notification[Flag_Directory]), prefix + "Expected directory but file is given.");
        pc("file_not_exist", std::weak_ptr<bool>(this->open_notification[Flag_NoExistenceCheck]), prefix + "Path does not exist.");
        std::string log_exts;
        for (auto& ext : this->extensions) {
            log_exts += "'." + ext + "' ";
        }
        pc("file_wrong_ext", std::weak_ptr<bool>(this->open_notification[Flag_RestrictExtension]), prefix + "File does not have required extension: " + log_exts);

        this->registered_notifications = true;
        return true;
    }
    return false;
}
