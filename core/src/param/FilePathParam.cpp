/*
 * FilePathParam.cpp
 *
 * Copyright (C) 2008 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "mmcore/param/FilePathParam.h"


using namespace megamol::core::param;


FilePathParam::FilePathParam(const std::filesystem::path& initVal, FilePathFlags flags)
        : AbstractParam(), flags(flags), val(initVal) {
    this->InitPresentation(AbstractParamPresentation::ParamType::FILEPATH);
}


FilePathParam::FilePathParam(const std::string& initVal, FilePathFlags flags)
        : AbstractParam(), flags(flags), val(initVal) {
    this->InitPresentation(AbstractParamPresentation::ParamType::FILEPATH);
}


FilePathParam::FilePathParam(const std::wstring& initVal, FilePathFlags flags)
        : AbstractParam(), flags(flags), val(initVal) {
    this->InitPresentation(AbstractParamPresentation::ParamType::FILEPATH);
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


void FilePathParam::SetValue(const std::filesystem::path& v, bool setDirty) {
    if (this->val != v) {
        this->val = v;
        this->indicateChange();
        if (setDirty) this->setDirty();
    }
}


void FilePathParam::SetValue(const std::string& v, bool setDirty) {
    this->SetValue(std::filesystem::path(v), setDirty);
}


void FilePathParam::SetValue(const std::wstring& v, bool setDirty) {
    this->SetValue(std::filesystem::path(v), setDirty);
}


vislib::TString FilePathParam::ValueString(void) const {
    return vislib::TString(this->val.generic_u8string().c_str());
}
