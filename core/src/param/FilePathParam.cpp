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
    , value() {

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
            }
            if (error_flags & Flag_Directory) {
                megamol::core::utility::log::Log::DefaultLog.WriteWarn(
                    "[FilePathParam] Omitting value '%s'. Expected directory but file is given.", new_value.generic_u8string().c_str());
            }
            if (error_flags & Flag_NoExistenceCheck) {
                megamol::core::utility::log::Log::DefaultLog.WriteWarn(
                    "[FilePathParam] Omitting value '%s'. File does not exist.", new_value.generic_u8string().c_str());
            }
            if (error_flags & Flag_RestrictExtension) {
                std::string log_exts;
                for (auto& ext : this->extensions) {
                    log_exts += "'." + ext + "' ";
                }
                megamol::core::utility::log::Log::DefaultLog.WriteWarn(
                    "[FilePathParam] Omitting value '%s'. File does not have required extension: %s", new_value.generic_u8string().c_str(), log_exts.c_str());
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
