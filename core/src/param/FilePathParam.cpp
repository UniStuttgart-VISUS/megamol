/*
 * FilePathParam.cpp
 *
 * Copyright (C) 2008 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "mmcore/param/FilePathParam.h"


using namespace megamol::core::param;


FilePathParam::FilePathParam(const std::string& initVal, Flags_t flags, const Extensions_t& exts)
    : AbstractParam()
    , flags(flags)
    , extensions(exts)
    , value(initVal)
    , registered_notifications(false)
    , open_notification() {

    this->InitPresentation(AbstractParamPresentation::ParamType::FILEPATH);

    this->open_notification[Flag_File] = std::make_shared<bool>(false);
    this->open_notification[Flag_Directory] = std::make_shared<bool>(false);
    this->open_notification[Flag_NoExistenceCheck] = std::make_shared<bool>(false);
    this->open_notification[Flag_RestrictExtension] = std::make_shared<bool>(false);
}


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


void FilePathParam::SetValue(const std::string& v, bool setDirty) {

    //auto new_value = std::filesystem::path(v);
    vislib::TString enc_tmp;
    vislib::UTF8Encoder::Encode(enc_tmp, vislib::TString(v.c_str()));
    auto new_value = std::filesystem::path(std::string(enc_tmp.PeekBuffer()));

    if (this->value != new_value) {
        auto error_flags = FilePathParam::ValidatePath(new_value, this->extensions, this->flags);

        if (error_flags & Flag_File) {
            megamol::core::utility::log::Log::DefaultLog.WriteWarn("[FilePathParam] Omitting new value '%s'. Expected file but directory is given.", new_value.c_str());
            if (this->open_notification[Flag_File] != nullptr)
                *this->open_notification[Flag_File] = true;
        }
        if (error_flags & Flag_Directory) {
            megamol::core::utility::log::Log::DefaultLog.WriteWarn("[FilePathParam] Omitting new value '%s'. Expected directory but file is given.", new_value.c_str());
            if (this->open_notification[Flag_Directory] != nullptr)
                *this->open_notification[Flag_Directory] = true;
        }
        if (error_flags & Flag_NoExistenceCheck) {
            megamol::core::utility::log::Log::DefaultLog.WriteWarn("[FilePathParam] Omitting new value '%s'. File does not exist.", new_value.c_str());
            if (this->open_notification[Flag_NoExistenceCheck] != nullptr)
                *this->open_notification[Flag_NoExistenceCheck] = true;
        }
        if (error_flags & Flag_RestrictExtension) {
            std::string log_exts;
            for (auto& ext : this->extensions) {
                log_exts += "'." + ext + "' ";
            }
            megamol::core::utility::log::Log::DefaultLog.WriteWarn(
                "[FilePathParam] Omitting new value '%s'. File does not have required extension: %s",
                new_value.c_str(), log_exts.c_str());
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
}


void FilePathParam::SetValue(const vislib::TString& v, bool setDirty) {

    this->SetValue(std::string(v.PeekBuffer()), setDirty);
}


FilePathParam::Flags_t FilePathParam::ValidatePath(const std::filesystem::path& p, const Extensions_t& e, Flags_t f) {

    FilePathParam::Flags_t retval = 0;
    if ((f & FilePathParam::Flag_File) && std::filesystem::is_directory(p)) { /// !std::filesystem::is_regular_file(p)
        retval |= FilePathParam::Flag_File;
    }
    if ((f & FilePathParam::Flag_Directory) && !std::filesystem::is_directory(p)) {
        retval |= FilePathParam::Flag_Directory;
    }
    if (!(f & Flag_NoExistenceCheck) && !std::filesystem::exists(p)) {
        retval |= FilePathParam::Flag_NoExistenceCheck;
    }
    if (f & FilePathParam::Flag_RestrictExtension) {
        bool valid_ext = false;
        for (auto& ext : e) {
            if (p.extension().generic_string() == std::string("." + ext)) {
                valid_ext = true;
            }
        }
        if (!valid_ext) {
            retval |= FilePathParam::Flag_RestrictExtension;
        }
    }
    return retval;
}


bool FilePathParam::RegisterNotifications(const FilePathParam::RegisterNotificationCallback_t& pc) {

    if (!this->registered_notifications) {
        const std::string prefix = "Omitting new value. ";
        pc("file_is_dir", this->open_notification[Flag_File], prefix + "Expected file but directory is given.");
        pc("dir_is_file", this->open_notification[Flag_Directory], prefix + "Expected directory but file is given.");
        pc("file_not_exist", this->open_notification[Flag_NoExistenceCheck], prefix + "Path does not exist.");
        std::string log_exts;
        for (auto& ext : this->extensions) {
            log_exts += "'." + ext + "' ";
        }
        pc("file_wrong_ext", this->open_notification[Flag_RestrictExtension], prefix + "File does not have required extension: " + log_exts);

        this->registered_notifications = true;
        return true;
    }
    return false;
}
