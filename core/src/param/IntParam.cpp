/*
 * IntParam.cpp
 *
 * Copyright (C) 2008 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "mmcore/param/IntParam.h"
#include <climits>

using namespace megamol::core::param;


/*
 * IntParam::IntParam
 */
IntParam::IntParam(int initVal)
        : AbstractParam(), val(initVal), minVal(INT_MIN),
        maxVal(INT_MAX) {
    ASSERT(this->minVal <= this->maxVal);
    ASSERT(this->minVal <= this->val);
    ASSERT(this->val <= this->maxVal);
}


/*
 * IntParam::IntParam
 */
IntParam::IntParam(int initVal, int minVal)
        : AbstractParam(), val(initVal), minVal(minVal),
        maxVal(INT_MAX) {
    ASSERT(this->minVal <= this->maxVal);
    ASSERT(this->minVal <= this->val);
    ASSERT(this->val <= this->maxVal);
}


/*
 * IntParam::IntParam
 */
IntParam::IntParam(int initVal, int minVal, int maxVal)
        : AbstractParam(), val(initVal), minVal(minVal),
        maxVal(maxVal) {
    ASSERT(this->minVal <= this->maxVal);
    ASSERT(this->minVal <= this->val);
    ASSERT(this->val <= this->maxVal);
}


/*
 * IntParam::~IntParam
 */
IntParam::~IntParam(void) {
    // intentionally empty
}


/*
 * IntParam::Definition
 */
void IntParam::Definition(vislib::RawStorage& outDef) const {
    outDef.AssertSize(6 + 2 * sizeof(int));
    memcpy(outDef.AsAt<char>(0), "MMINTR", 6);
    *outDef.AsAt<int>(6) = this->minVal;
    *outDef.AsAt<int>(6 + sizeof(int)) = this->maxVal;
}


/*
 * IntParam::ParseValue
 */
bool IntParam::ParseValue(const vislib::TString& v) {
    try {
        this->SetValue(vislib::TCharTraits::ParseInt(v));
        return true;
    } catch(...) {
    }
    return false;
}


/*
 * IntParam::SetValue
 */
void IntParam::SetValue(int v, bool setDirty) {
    if (v < this->minVal) v = this->minVal;
    else if (v > this->maxVal) v = this->maxVal;
    if (this->val != v) {
        this->val = v;
        this->indicateChange();
        if (setDirty) this->setDirty();
    }
}


/*
 * IntParam::ValueString
 */
vislib::TString IntParam::ValueString(void) const {
    vislib::TString str;
    str.Format(_T("%d"), this->val);
    return str;
}
