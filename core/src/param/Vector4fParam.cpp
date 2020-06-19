/*
 * Vector4fParam.cpp
 *
 * Copyright (C) 2009, 2020 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "mmcore/param/Vector4fParam.h"
#include <cfloat>
#include <sstream>
#include "vislib/StringTokeniser.h"
#include "vislib/assert.h"

using namespace megamol::core;


/*
 * param::Vector4fParam::Vector4fParam
 */
param::Vector4fParam::Vector4fParam(const vislib::math::Vector<float, 4>& initVal)
    : AbstractParam()
    , val(initVal)
    , minVal(-FLT_MAX, -FLT_MAX, -FLT_MAX, -FLT_MAX)
    , maxVal(FLT_MAX, FLT_MAX, FLT_MAX, FLT_MAX) {
    ASSERT(this->isLessOrEqual(this->minVal, this->maxVal));
    ASSERT(this->isLessOrEqual(this->minVal, this->val));
    ASSERT(this->isLessOrEqual(this->val, this->maxVal));
}


/*
 * param::Vector4fParam::Vector4fParam
 */
param::Vector4fParam::Vector4fParam(
    const vislib::math::Vector<float, 4>& initVal, const vislib::math::Vector<float, 4>& minVal)
    : AbstractParam(), val(initVal), minVal(minVal), maxVal(FLT_MAX, FLT_MAX, FLT_MAX, FLT_MAX) {
    ASSERT(this->isLessOrEqual(this->minVal, this->maxVal));
    ASSERT(this->isLessOrEqual(this->minVal, this->val));
    ASSERT(this->isLessOrEqual(this->val, this->maxVal));
}


/*
 * param::Vector4fParam::Vector4fParam
 */
param::Vector4fParam::Vector4fParam(const vislib::math::Vector<float, 4>& initVal,
    const vislib::math::Vector<float, 4>& minVal, const vislib::math::Vector<float, 4>& maxVal)
    : AbstractParam(), val(initVal), minVal(minVal), maxVal(maxVal) {
    ASSERT(this->isLessOrEqual(this->minVal, this->maxVal));
    ASSERT(this->isLessOrEqual(this->minVal, this->val));
    ASSERT(this->isLessOrEqual(this->val, this->maxVal));
}


/*
 * param::Vector4fParam::~Vector4fParam
 */
param::Vector4fParam::~Vector4fParam(void) {
    // intentionally empty
}


/*
 * param::Vector4fParam::Definition
 */
void param::Vector4fParam::Definition(vislib::RawStorage& outDef) const {
    outDef.AssertSize(6 + 8 * sizeof(float));

    memcpy(outDef.As<char>(), "MMVC4F", 6);
    *outDef.AsAt<float>(6) = this->minVal[0];
    *outDef.AsAt<float>(6 + sizeof(float)) = this->minVal[1];
    *outDef.AsAt<float>(6 + 2 * sizeof(float)) = this->minVal[2];
    *outDef.AsAt<float>(6 + 3 * sizeof(float)) = this->minVal[3];
    *outDef.AsAt<float>(6 + 4 * sizeof(float)) = this->maxVal[0];
    *outDef.AsAt<float>(6 + 5 * sizeof(float)) = this->maxVal[1];
    *outDef.AsAt<float>(6 + 6 * sizeof(float)) = this->maxVal[2];
    *outDef.AsAt<float>(6 + 7 * sizeof(float)) = this->maxVal[3];
}


/*
 * param::Vector4fParam::ParseValue
 */
bool param::Vector4fParam::ParseValue(const vislib::TString& v) {
    vislib::Array<vislib::TString> comps = vislib::TStringTokeniser::Split(v, _T(";"), true);
    if (comps.Count() == 4) {
        try {
            comps[0].TrimSpaces();
            comps[1].TrimSpaces();
            comps[2].TrimSpaces();
            comps[3].TrimSpaces();
            float x = static_cast<float>(vislib::TCharTraits::ParseDouble(comps[0]));
            float y = static_cast<float>(vislib::TCharTraits::ParseDouble(comps[1]));
            float z = static_cast<float>(vislib::TCharTraits::ParseDouble(comps[2]));
            float w = static_cast<float>(vislib::TCharTraits::ParseDouble(comps[3]));

            this->SetValue(vislib::math::Vector<float, 4>(x, y, z, w));
            return true;
        } catch (...) {
        }
    }
    return false;
}


/*
 * param::Vector4fParam::SetValue
 */
void param::Vector4fParam::SetValue(const vislib::math::Vector<float, 4>& v, bool setDirty) {
    if (this->isLessOrEqual(v, this->minVal)) {
        if (this->val != this->minVal) {
            this->val = this->minVal;
            this->indicateChange();
            if (setDirty) this->setDirty();
        }
    } else if (this->isGreaterOrEqual(v, this->maxVal)) {
        if (this->val != this->maxVal) {
            this->val = this->maxVal;
            this->indicateChange();
            if (setDirty) this->setDirty();
        }
    } else if (this->val != v) {
        this->val = v;
        this->indicateChange();
        if (setDirty) this->setDirty();
    }
}


/*
 * param::Vector4fParam::ValueString
 */
vislib::TString param::Vector4fParam::ValueString(void) const {
    std::stringstream stream;
    stream.precision(std::numeric_limits<float>::max_digits10);
    stream << this->val[0] << ";" << this->val[1] << ";" << this->val[2] << ";" << this->val[3];
    return stream.str().c_str();
}


/*
 * param::Vector4fParam::isLessOrEqual
 */
bool param::Vector4fParam::isLessOrEqual(
    const vislib::math::Vector<float, 4>& A, const vislib::math::Vector<float, 4>& B) const {
    for (int i = 0; i < 4; i++) {
        if (A[i] > B[i]) {
            return false;
        }
    }
    return true;
}


/*
 * param::Vector4fParam::isGreaterOrEqual
 */
bool param::Vector4fParam::isGreaterOrEqual(
    const vislib::math::Vector<float, 4>& A, const vislib::math::Vector<float, 4>& B) const {
    for (int i = 0; i < 4; i++) {
        if (A[i] < B[i]) {
            return false;
        }
    }
    return true;
}
