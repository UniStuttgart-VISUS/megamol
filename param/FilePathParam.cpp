/*
 * FilePathParam.cpp
 *
 * Copyright (C) 2008 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "FilePathParam.h"
#include "vislib/StringConverter.h"

using namespace megamol::core::param;


/*
 * FilePathParam::FilePathParam
 */
FilePathParam::FilePathParam(const vislib::TString& initVal)
        : AbstractParam(), val(initVal) {
    // intentionally empty
}


/*
 * FilePathParam::FilePathParam
 */
FilePathParam::FilePathParam(const char *initVal)
        : AbstractParam(), val(initVal) {
    // intentionally empty
}


/*
 * FilePathParam::FilePathParam
 */
FilePathParam::FilePathParam(const wchar_t *initVal)
        : AbstractParam(), val(initVal) {
    // intentionally empty
}


/*
 * FilePathParam::~FilePathParam
 */
FilePathParam::~FilePathParam(void) {
    // intentionally empty
}


/*
 * FilePathParam::Definition
 */
void FilePathParam::Definition(vislib::RawStorage& outDef) const {
    outDef.AssertSize(6);
#if defined(UNICODE) || defined(_UNICODE)
    memcpy(outDef.AsAt<char>(0), "MMFILW", 6);
#else  /* defined(UNICODE) || defined(_UNICODE) */
    memcpy(outDef.AsAt<char>(0), "MMFILA", 6);
#endif /* defined(UNICODE) || defined(_UNICODE) */
}


/*
 * FilePathParam::ParseValue
 */
bool FilePathParam::ParseValue(const vislib::TString& v) {
    try {
        this->SetValue(v);
        return true;
    } catch(...) {
    }
    return false;
}


/*
 * FilePathParam::SetValue
 */
void FilePathParam::SetValue(const vislib::TString& v, bool setDirty) {
    if (this->val != v) {
        this->val = v;
        if (setDirty) this->setDirty();
    }
}


/*
 * FilePathParam::SetValue
 */
void FilePathParam::SetValue(const char *v, bool setDirty) {
    if (!this->val.Equals(A2T(v))) {
        this->val = v;
        if (setDirty) this->setDirty();
    }
}


/*
 * FilePathParam::SetValue
 */
void FilePathParam::SetValue(const wchar_t *v, bool setDirty) {
    if (!this->val.Equals(W2T(v))) {
        this->val = v;
        if (setDirty) this->setDirty();
    }
}


/*
 * FilePathParam::ValueString
 */
vislib::TString FilePathParam::ValueString(void) const {
    return this->val;
}
