/*
 * StringParam.cpp
 *
 * Copyright (C) 2008 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "mmcore/param/StringParam.h"
#include "vislib/StringConverter.h"

using namespace megamol::core::param;


/*
 * StringParam::StringParam
 */
StringParam::StringParam(const vislib::StringA& initVal)
        : AbstractParam(), val(initVal) {
    // intentionally empty
}


/*
 * StringParam::StringParam
 */
StringParam::StringParam(const vislib::StringW& initVal)
        : AbstractParam(), val(initVal) {
    // intentionally empty
}


/*
 * StringParam::StringParam
 */
StringParam::StringParam(const char *initVal)
        : AbstractParam(), val(initVal) {
    // intentionally empty
}


/*
 * StringParam::StringParam
 */
StringParam::StringParam(const wchar_t *initVal)
        : AbstractParam(), val(initVal) {
    // intentionally empty
}


/*
 * StringParam::~StringParam
 */
StringParam::~StringParam(void) {
    // intentionally empty
}


/*
 * StringParam::Definition
 */
void StringParam::Definition(vislib::RawStorage& outDef) const {
    outDef.AssertSize(6);
#if defined(UNICODE) || defined(_UNICODE)
    memcpy(outDef.AsAt<char>(0), "MMSTRW", 6);
#else  /* defined(UNICODE) || defined(_UNICODE) */
    memcpy(outDef.AsAt<char>(0), "MMSTRA", 6);
#endif /* defined(UNICODE) || defined(_UNICODE) */
}


/*
 * StringParam::ParseValue
 */
bool StringParam::ParseValue(const vislib::TString& v) {
    try {
        this->SetValue(v);
        return true;
    } catch(...) {
    }
    return false;
}


/*
 * StringParam::SetValue
 */
void StringParam::SetValue(const vislib::StringA& v, bool setDirty) {
    if (this->val != v) {
        this->val = v;
        this->indicateChange();
        if (setDirty) this->setDirty();
    }
}


/*
 * StringParam::SetValue
 */
void StringParam::SetValue(const vislib::StringW& v, bool setDirty) {
    if (this->val != v) {
        this->val = v;
        this->indicateChange();
        if (setDirty) this->setDirty();
    }
}


/*
 * StringParam::SetValue
 */
void StringParam::SetValue(const char *v, bool setDirty) {
    if (!this->val.Equals(A2T(v))) {
        this->val = v;
        this->indicateChange();
        if (setDirty) this->setDirty();
    }
}


/*
 * StringParam::SetValue
 */
void StringParam::SetValue(const wchar_t *v, bool setDirty) {
    if (!this->val.Equals(W2T(v))) {
        this->val = v;
        this->indicateChange();
        if (setDirty) this->setDirty();
    }
}


/*
 * StringParam::ValueString
 */
vislib::TString StringParam::ValueString(void) const {
    return this->val;
}
