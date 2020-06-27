/*
 * FloatParam.cpp
 *
 * Copyright (C) 2008 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "mmcore/param/FloatParam.h"
#include <cfloat>

using namespace megamol::core::param;


/*
 * FloatParam::FloatParam
 */
FloatParam::FloatParam(float initVal)
        : AbstractParam(), val(initVal), minVal(-FLT_MAX),
        maxVal(FLT_MAX) {
    ASSERT(this->minVal <= this->maxVal);
    ASSERT(this->minVal <= this->val);
    ASSERT(this->val <= this->maxVal);
}


/*
 * FloatParam::FloatParam
 */
FloatParam::FloatParam(float initVal, float minVal)
        : AbstractParam(), val(initVal), minVal(minVal),
        maxVal(FLT_MAX) {
    ASSERT(this->minVal <= this->maxVal);
    ASSERT(this->minVal <= this->val);
    ASSERT(this->val <= this->maxVal);
}


/*
 * FloatParam::FloatParam
 */
FloatParam::FloatParam(float initVal, float minVal, float maxVal)
        : AbstractParam(), val(initVal), minVal(minVal), maxVal(maxVal) {
    ASSERT(this->minVal <= this->maxVal);
    ASSERT(this->minVal <= this->val);
    ASSERT(this->val <= this->maxVal);
}


/*
 * FloatParam::~FloatParam
 */
FloatParam::~FloatParam(void) {
    // intentionally empty
}


/*
 * FloatParam::Definition
 */
void FloatParam::Definition(vislib::RawStorage& outDef) const {
    outDef.AssertSize(14);
    memcpy(outDef.AsAt<char>(0), "MMFLOT", 6);
    *outDef.AsAt<float>(6) = this->minVal;
    *outDef.AsAt<float>(10) = this->maxVal;
}


/*
 * FloatParam::ParseValue
 */
bool FloatParam::ParseValue(const vislib::TString& v) {
    try {
        this->SetValue((float)vislib::TCharTraits::ParseDouble(v));
        return true;
    } catch(...) {
    }
    return false;
}


/*
 * FloatParam::SetValue
 */
void FloatParam::SetValue(float v, bool setDirty) {
    if (v < this->minVal) v = this->minVal;
    else if (v > this->maxVal) v = this->maxVal;
    if (this->val != v) {
        this->val = v;
        this->indicateChange();
        if (setDirty) this->setDirty();
    }
}


/*
 * FloatParam::ValueString
 */
vislib::TString FloatParam::ValueString(void) const {
    vislib::TString str;
    str.Format(_T("%f"), this->val);
    return str;
}
