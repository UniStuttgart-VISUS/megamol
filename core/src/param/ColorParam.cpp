/*
 * StringParam.cpp
 *
 * Copyright (C) 2008 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "mmcore/param/ColorParam.h"
#include "vislib/StringConverter.h"


megamol::core::param::ColorParam::ColorParam(vislib::StringA const& initVal) {
}


megamol::core::param::ColorParam::ColorParam(vislib::StringW const& initVal) {

}


megamol::core::param::ColorParam::ColorParam(char const* initVal) {

}


megamol::core::param::ColorParam::ColorParam(wchar_t const* initVal) {
}


megamol::core::param::ColorParam::~ColorParam(void) {

}


void megamol::core::param::ColorParam::Definition(vislib::RawStorage& outDef) const {
    outDef.AssertSize(6);
#if defined(UNICODE) || defined(_UNICODE)
    memcpy(outDef.AsAt<char>(0), "MMTF1W", 6);
#else  /* defined(UNICODE) || defined(_UNICODE) */
    memcpy(outDef.AsAt<char>(0), "MMTF1A", 6);
#endif /* defined(UNICODE) || defined(_UNICODE) */
}


bool megamol::core::param::ColorParam::ParseValue(vislib::TString const& v) {
    try {
        this->SetValue(v);
        return true;
    } catch (...) {
    }
    return false;
}


void megamol::core::param::ColorParam::SetValue(vislib::StringA const& v, bool setDirty) {
    if (this->val != v) {
        this->val = v;
        if (setDirty) this->setDirty();
    }
}


void megamol::core::param::ColorParam::SetValue(vislib::StringW const& v, bool setDirty) {
    if (this->val != v) {
        this->val = v;
        if (setDirty) this->setDirty();
    }
}


void megamol::core::param::ColorParam::SetValue(char const* v, bool setDirty) {
    if (!this->val.Equals(A2T(v))) {
        this->val = v;
        if (setDirty) this->setDirty();
    }
}


void megamol::core::param::ColorParam::SetValue(wchar_t const* v, bool setDirty) {
    if (!this->val.Equals(W2T(v))) {
        this->val = v;
        if (setDirty) this->setDirty();
    }
}


vislib::TString megamol::core::param::ColorParam::ValueString(void) const {
    return this->val;
}
