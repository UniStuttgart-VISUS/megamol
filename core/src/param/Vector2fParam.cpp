/*
 * Vector2fParam.cpp
 *
 * Copyright (C) 2009, 2020 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "mmcore/param/Vector2fParam.h"
#include <cfloat>
#include <sstream>
#include "vislib/StringTokeniser.h"
#include "vislib/assert.h"

using namespace megamol::core;


/*
 * param::Vector2fParam::Vector2fParam
 */
param::Vector2fParam::Vector2fParam(const vislib::math::Vector<float, 2>& initVal)
    : AbstractParam(), val(initVal), minVal(-FLT_MAX, -FLT_MAX), maxVal(FLT_MAX, FLT_MAX) {
    ASSERT(this->isLessOrEqual(this->minVal, this->maxVal));
    ASSERT(this->isLessOrEqual(this->minVal, this->val));
    ASSERT(this->isLessOrEqual(this->val, this->maxVal));
}


/*
 * param::Vector2fParam::Vector2fParam
 */
param::Vector2fParam::Vector2fParam(
    const vislib::math::Vector<float, 2>& initVal, const vislib::math::Vector<float, 2>& minVal)
    : AbstractParam(), val(initVal), minVal(minVal), maxVal(FLT_MAX, FLT_MAX) {
    ASSERT(this->isLessOrEqual(this->minVal, this->maxVal));
    ASSERT(this->isLessOrEqual(this->minVal, this->val));
    ASSERT(this->isLessOrEqual(this->val, this->maxVal));
}


/*
 * param::Vector2fParam::Vector2fParam
 */
param::Vector2fParam::Vector2fParam(const vislib::math::Vector<float, 2>& initVal,
    const vislib::math::Vector<float, 2>& minVal, const vislib::math::Vector<float, 2>& maxVal)
    : AbstractParam(), val(initVal), minVal(minVal), maxVal(maxVal) {
    ASSERT(this->isLessOrEqual(this->minVal, this->maxVal));
    ASSERT(this->isLessOrEqual(this->minVal, this->val));
    ASSERT(this->isLessOrEqual(this->val, this->maxVal));
}


/*
 * param::Vector2fParam::~Vector2fParam
 */
param::Vector2fParam::~Vector2fParam(void) {
    // intentionally empty
}


/*
 * param::Vector2fParam::Definition
 */
void param::Vector2fParam::Definition(vislib::RawStorage& outDef) const {
    outDef.AssertSize(6 + 4 * sizeof(float));

    memcpy(outDef.As<char>(), "MMVC2F", 6);
    *outDef.AsAt<float>(6) = this->minVal[0];
    *outDef.AsAt<float>(6 + sizeof(float)) = this->minVal[1];
    *outDef.AsAt<float>(6 + 2 * sizeof(float)) = this->maxVal[0];
    *outDef.AsAt<float>(6 + 3 * sizeof(float)) = this->maxVal[1];
}


/*
 * param::Vector2fParam::ParseValue
 */
bool param::Vector2fParam::ParseValue(const vislib::TString& v) {
    vislib::Array<vislib::TString> comps = vislib::TStringTokeniser::Split(v, _T(";"), true);
    if (comps.Count() == 2) {
        try {
            comps[0].TrimSpaces();
            comps[1].TrimSpaces();
            float x = static_cast<float>(vislib::TCharTraits::ParseDouble(comps[0]));
            float y = static_cast<float>(vislib::TCharTraits::ParseDouble(comps[1]));

            this->SetValue(vislib::math::Vector<float, 2>(x, y));
            return true;
        } catch (...) {
        }
    }
    return false;
}


/*
 * param::Vector2fParam::SetValue
 */
void param::Vector2fParam::SetValue(const vislib::math::Vector<float, 2>& v, bool setDirty) {
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
 * param::Vector2fParam::ValueString
 */
vislib::TString param::Vector2fParam::ValueString(void) const {
    std::stringstream stream;
    stream.precision(std::numeric_limits<float>::max_digits10);
    stream << this->val[0] << ";" << this->val[1];
    return stream.str().c_str();
}


/*
 * param::Vector2fParam::isLessOrEqual
 */
bool param::Vector2fParam::isLessOrEqual(
    const vislib::math::Vector<float, 2>& A, const vislib::math::Vector<float, 2>& B) const {
    for (int i = 0; i < 2; i++) {
        if (A[i] > B[i]) {
            return false;
        }
    }
    return true;
}


/*
 * param::Vector2fParam::isGreaterOrEqual
 */
bool param::Vector2fParam::isGreaterOrEqual(
    const vislib::math::Vector<float, 2>& A, const vislib::math::Vector<float, 2>& B) const {
    for (int i = 0; i < 2; i++) {
        if (A[i] < B[i]) {
            return false;
        }
    }
    return true;
}
