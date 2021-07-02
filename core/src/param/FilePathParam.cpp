/*
 * FilePathParam.cpp
 *
 * Copyright (C) 2008 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "mmcore/param/FilePathParam.h"


using namespace megamol::core::param;


FilePathParam::FilePathParam(const std::string& initVal, FilePathFlags_t flags, FilePathExtensions_t exts)
        : AbstractParam(), value(initVal), flags(flags), extensions(exts) {

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


void FilePathParam::SetValue(const std::string& v, bool setDirty) {

    if (this->valid_change(v)) {
        this->value = static_cast<std::filesystem::path>(v);
        this->indicateChange();
        if (setDirty) this->setDirty();
    }
}


void FilePathParam::SetValue(const vislib::TString& v, bool setDirty) {

    this->SetValue(std::string(v.PeekBuffer()), setDirty);
}


bool FilePathParam::valid_change(std::string v) {

    auto new_value = static_cast<std::filesystem::path>(v);
    if (this->value != new_value) {
        return true;
    }
    return false;
}
