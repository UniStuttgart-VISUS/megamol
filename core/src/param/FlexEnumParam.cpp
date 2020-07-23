/*
 * FlexEnumParam.cpp
 *
 * Copyright (C) 2017 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "mmcore/param/FlexEnumParam.h"
#include "vislib/IllegalStateException.h"
#include "vislib/StringConverter.h"
#include "vislib/UTF8Encoder.h"
#include <set>
#include <string>

using namespace megamol::core::param;


/*
 * FlexEnumParam::FlexEnumParam
 */
FlexEnumParam::FlexEnumParam(const std::string& initVal)
        : AbstractParam(), val(initVal), values() {
    // intentionally empty
}


/*
 * FlexEnumParam::~FlexEnumParam
 */
FlexEnumParam::~FlexEnumParam(void) {
    // intentionally empty
}


/*
 * FlexEnumParam::ClearTypePairs
 */
void megamol::core::param::FlexEnumParam::ClearValues(void) {
    this->SetHash(this->GetHash() + 1);
    this->values.clear();
}


/*
 * FlexEnumParam::Definition
 */
void FlexEnumParam::Definition(vislib::RawStorage& outDef) const {
    vislib::StringA utf8;
    unsigned int s = 6;
    unsigned int c = 0;
    for (const auto &v : this->values) {
        s += static_cast<unsigned int>(v.length() + 1); //terminating zero
    }
    s += sizeof(unsigned int);

    outDef.AssertSize(s);
    memcpy(outDef.AsAt<char>(0), "MMFENU", 6);
    s = 6 + sizeof(unsigned int);
    for (const auto &v: this->values) {
        unsigned int strsize = static_cast<unsigned int>(v.length() + 1);
        memcpy(outDef.AsAt<char>(s), v.c_str(), strsize);
        s += strsize;
        c++;
    }
    *outDef.AsAt<unsigned int>(6) = c;
}


/*
 * FlexEnumParam::ParseValue
 */
bool FlexEnumParam::ParseValue(const vislib::TString& v) {
    try {
        this->SetValue(std::string(T2A(v)));
        return true;
        //auto iter = this->values.find(std::string(T2A(v)));
        //if (iter == this->values.end()) {
        //    return false;
        //} else {
        //    this->SetValue(*iter);
        //    return true;
        //}
    } catch(...) {
    }
    return false;
}


/*
 * FlexEnumParam::AddValue
 */
FlexEnumParam* FlexEnumParam::AddValue(const std::string& name) {
    auto iter = this->values.find(name);
    if (iter == this->values.end()) {
        this->SetHash(this->GetHash() + 1);
        this->values.insert(name);
    }
    return this;
}


/*
 * FlexEnumParam::SetValue
 */
void FlexEnumParam::SetValue(const std::string& v, bool setDirty) {
    if (this->val != v) {
        this->val = v;
        this->indicateChange();
        if (setDirty) this->setDirty();
    }
}


/*
 * FlexEnumParam::ValueString
 */
vislib::TString FlexEnumParam::ValueString(void) const {
    return A2T(val.c_str());
}
