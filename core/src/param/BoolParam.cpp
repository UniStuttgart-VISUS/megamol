/*
 * BoolParam.cpp
 *
 * Copyright (C) 2008 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "mmcore/param/BoolParam.h"

using namespace megamol::core::param;


/*
 * BoolParam::BoolParam
 */
BoolParam::BoolParam(bool initVal)
        : AbstractParam(), val(initVal) {
    // intentionally empty
}


/*
 * BoolParam::~BoolParam
 */
BoolParam::~BoolParam(void) {
    // intentionally empty
}


/*
 * BoolParam::Definition
 */
void BoolParam::Definition(vislib::RawStorage& outDef) const {
    outDef.AssertSize(6);
    memcpy(outDef.AsAt<char>(0), "MMBOOL", 6);
}


/*
 * BoolParam::ParseValue
 */
bool BoolParam::ParseValue(const vislib::TString& v) {
    try {
        this->SetValue(vislib::TCharTraits::ParseBool(v));
        return true;
    } catch(...) {
    }
    return false;
}


/*
 * BoolParam::SetValue
 */
void BoolParam::SetValue(bool v, bool setDirty) {
    if (this->val != v) {
        this->val = v;
        this->indicateChange();
        if (setDirty) this->setDirty();
    }
}


/*
 * BoolParam::ValueString
 */
vislib::TString BoolParam::ValueString(void) const {
    return this->val ? _T("true") : _T("false");
}
