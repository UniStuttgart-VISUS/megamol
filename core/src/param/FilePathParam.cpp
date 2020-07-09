/*
 * FilePathParam.cpp
 *
 * Copyright (C) 2008 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "mmcore/param/FilePathParam.h"
#include "vislib/StringConverter.h"

using namespace megamol::core::param;


/*
 * FilePathParam::FLAG_NONE
 */
const UINT32 FilePathParam::FLAG_NONE = 0x00000000;


/*
 * FilePathParam::FLAG_NOPATHCHANGE
 */
const UINT32 FilePathParam::FLAG_NOPATHCHANGE = 0x00000001;


/*
 * FilePathParam::FLAG_NOEXISTANCECHECK
 */
const UINT32 FilePathParam::FLAG_NOEXISTANCECHECK = 0x00000002;


/*
 * FilePathParam::FLAG_TOBECREATED
 */
const UINT32 FilePathParam::FLAG_TOBECREATED = 0x00000003;


/*
 * FilePathParam::FilePathParam
 */
FilePathParam::FilePathParam(const vislib::StringA& initVal, UINT32 flags)
        : AbstractParam(), flags(flags), val(initVal) {
    // intentionally empty
}


/*
 * FilePathParam::FilePathParam
 */
FilePathParam::FilePathParam(const vislib::StringW& initVal, UINT32 flags)
        : AbstractParam(), flags(flags), val(initVal) {
    // intentionally empty
}


/*
 * FilePathParam::FilePathParam
 */
FilePathParam::FilePathParam(const char *initVal, UINT32 flags)
        : AbstractParam(), flags(flags), val(initVal) {
    // intentionally empty
}


/*
 * FilePathParam::FilePathParam
 */
FilePathParam::FilePathParam(const wchar_t *initVal, UINT32 flags)
        : AbstractParam(), flags(flags), val(initVal) {
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
void FilePathParam::SetValue(const vislib::StringA& v, bool setDirty) {
    if (this->val != v) {
        this->val = v;
        this->indicateChange();
        if (setDirty) this->setDirty();
    }
}


/*
 * FilePathParam::SetValue
 */
void FilePathParam::SetValue(const vislib::StringW& v, bool setDirty) {
    if (this->val != v) {
        this->val = v;
        this->indicateChange();
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
        this->indicateChange();
        if (setDirty) this->setDirty();
    }
}


/*
 * FilePathParam::ValueString
 */
vislib::TString FilePathParam::ValueString(void) const {
    return this->val;
}
