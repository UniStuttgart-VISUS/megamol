/*
 * TernaryParam.cpp
 *
 * Copyright (C) 2008 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "mmcore/param/TernaryParam.h"

using namespace megamol::core::param;


/*
 * TernaryParam::TernaryParam
 */
TernaryParam::TernaryParam(const vislib::math::Ternary& initVal)
        : AbstractParam(), val(initVal) {
    // intentionally empty
}


/*
 * TernaryParam::~TernaryParam
 */
TernaryParam::~TernaryParam(void) {
    // intentionally empty
}


/*
 * TernaryParam::Definition
 */
void TernaryParam::Definition(vislib::RawStorage& outDef) const {
    outDef.AssertSize(6);
    memcpy(outDef.AsAt<char>(0), "MMTRRY", 6);
}


/*
 * TernaryParam::ParseValue
 */
bool TernaryParam::ParseValue(const vislib::TString& v) {
    return this->val.Parse(v);
}


/*
 * TernaryParam::SetValue
 */
void TernaryParam::SetValue(vislib::math::Ternary v, bool setDirty) {
    if (this->val != v) {
        this->val = v;
        this->indicateChange();
        if (setDirty) this->setDirty();
    }
}


/*
 * TernaryParam::ValueString
 */
vislib::TString TernaryParam::ValueString(void) const {
#if defined(UNICODE) || defined(_UNICODE)
    return this->val.ToStringW();
#else /* defined(UNICODE) || defined(_UNICODE) */
    return this->val.ToStringA();
#endif /* defined(UNICODE) || defined(_UNICODE) */
}
