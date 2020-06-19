/*
 * Vector3fParam.cpp
 *
 * Copyright (C) 2009, 2020 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "mmcore/param/Vector3fParam.h"
#include <cfloat>
#include <sstream>
#include "vislib/StringTokeniser.h"
#include "vislib/assert.h"

using namespace megamol::core;


/*
 * param::Vector3fParam::Vector3fParam
 */
param::Vector3fParam::Vector3fParam(const vislib::math::Vector<float, 3>& initVal)
    : AbstractParam(), val(initVal), minVal(-FLT_MAX, -FLT_MAX, -FLT_MAX), maxVal(FLT_MAX, FLT_MAX, FLT_MAX) {
    ASSERT(this->isLessOrEqual(this->minVal, this->maxVal));
    ASSERT(this->isLessOrEqual(this->minVal, this->val));
    ASSERT(this->isLessOrEqual(this->val, this->maxVal));
}


/*
 * param::Vector3fParam::Vector3fParam
 */
param::Vector3fParam::Vector3fParam(
    const vislib::math::Vector<float, 3>& initVal, const vislib::math::Vector<float, 3>& minVal)
    : AbstractParam(), val(initVal), minVal(minVal), maxVal(FLT_MAX, FLT_MAX, FLT_MAX) {
    ASSERT(this->isLessOrEqual(this->minVal, this->maxVal));
    ASSERT(this->isLessOrEqual(this->minVal, this->val));
    ASSERT(this->isLessOrEqual(this->val, this->maxVal));
}


/*
 * param::Vector3fParam::Vector3fParam
 */
param::Vector3fParam::Vector3fParam(const vislib::math::Vector<float, 3>& initVal,
    const vislib::math::Vector<float, 3>& minVal, const vislib::math::Vector<float, 3>& maxVal)
    : AbstractParam(), val(initVal), minVal(minVal), maxVal(maxVal) {
    ASSERT(this->isLessOrEqual(this->minVal, this->maxVal));
    ASSERT(this->isLessOrEqual(this->minVal, this->val));
    ASSERT(this->isLessOrEqual(this->val, this->maxVal));
}


/*
 * param::Vector3fParam::~Vector3fParam
 */
param::Vector3fParam::~Vector3fParam(void) {
    // intentionally empty
}


/*
 * param::Vector3fParam::Definition
 */
void param::Vector3fParam::Definition(vislib::RawStorage& outDef) const {
    outDef.AssertSize(6 + 6 * sizeof(float));

    memcpy(outDef.As<char>(), "MMVC3F", 6);
    *outDef.AsAt<float>(6) = this->minVal[0];
    *outDef.AsAt<float>(6 + sizeof(float)) = this->minVal[1];
    *outDef.AsAt<float>(6 + 2 * sizeof(float)) = this->minVal[2];
    *outDef.AsAt<float>(6 + 3 * sizeof(float)) = this->maxVal[0];
    *outDef.AsAt<float>(6 + 4 * sizeof(float)) = this->maxVal[1];
    *outDef.AsAt<float>(6 + 5 * sizeof(float)) = this->maxVal[2];
}


/*
 * param::Vector3fParam::ParseValue
 */
bool param::Vector3fParam::ParseValue(const vislib::TString& v) {
    vislib::Array<vislib::TString> comps = vislib::TStringTokeniser::Split(v, _T(";"), true);
    if (comps.Count() == 3) {
        try {
            comps[0].TrimSpaces();
            comps[1].TrimSpaces();
            comps[2].TrimSpaces();
            float x = static_cast<float>(vislib::TCharTraits::ParseDouble(comps[0]));
            float y = static_cast<float>(vislib::TCharTraits::ParseDouble(comps[1]));
            float z = static_cast<float>(vislib::TCharTraits::ParseDouble(comps[2]));

            this->SetValue(vislib::math::Vector<float, 3>(x, y, z));
            return true;
        } catch (...) {
        }
    }
    return false;
}


/*
 * param::Vector3fParam::SetValue
 */
void param::Vector3fParam::SetValue(const vislib::math::Vector<float, 3>& v, bool setDirty) {
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
 * param::Vector3fParam::ValueString
 */
vislib::TString param::Vector3fParam::ValueString(void) const {
    std::stringstream stream;
    stream.precision(std::numeric_limits<float>::max_digits10);
    stream << this->val[0] << ";" << this->val[1] << ";" << this->val[2];
    return stream.str().c_str();
}


/*
 * param::Vector3fParam::isLessOrEqual
 */
bool param::Vector3fParam::isLessOrEqual(
    const vislib::math::Vector<float, 3>& A, const vislib::math::Vector<float, 3>& B) const {
    for (int i = 0; i < 3; i++) {
        if (A[i] > B[i]) {
            return false;
        }
    }
    return true;
}


/*
 * param::Vector3fParam::isGreaterOrEqual
 */
bool param::Vector3fParam::isGreaterOrEqual(
    const vislib::math::Vector<float, 3>& A, const vislib::math::Vector<float, 3>& B) const {
    for (int i = 0; i < 3; i++) {
        if (A[i] < B[i]) {
            return false;
        }
    }
    return true;
}
