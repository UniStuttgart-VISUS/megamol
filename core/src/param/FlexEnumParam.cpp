/*
 * FlexEnumParam.cpp
 *
 * Copyright (C) 2017 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

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
FlexEnumParam::FlexEnumParam(const std::string& initVal) : AbstractParam(), val(), values() {
    this->InitPresentation(AbstractParamPresentation::ParamType::FLEXENUM);
    this->SetValue(initVal);
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
std::string FlexEnumParam::Definition() const {
    vislib::StringA utf8;
    unsigned int s = 6;
    unsigned int c = 0;
    for (const auto& v : this->values) {
        s += static_cast<unsigned int>(v.length() + 1); //terminating zero
    }
    s += sizeof(unsigned int);

    vislib::RawStorage outDef;
    outDef.AssertSize(s);
    memcpy(outDef.AsAt<char>(0), "MMFENU", 6);
    s = 6 + sizeof(unsigned int);
    for (const auto& v : this->values) {
        unsigned int strsize = static_cast<unsigned int>(v.length() + 1);
        memcpy(outDef.AsAt<char>(s), v.c_str(), strsize);
        s += strsize;
        c++;
    }
    *outDef.AsAt<unsigned int>(6) = c;

    std::string return_str;
    return_str.resize(outDef.GetSize());
    std::copy(outDef.AsAt<char>(0), outDef.AsAt<char>(0) + outDef.GetSize(), return_str.begin());
    return return_str;
}


/*
 * FlexEnumParam::ParseValue
 */
bool FlexEnumParam::ParseValue(std::string const& v) {
    try {
        this->SetValue(v);
        return true;
        //auto iter = this->values.find(std::string(T2A(v)));
        //if (iter == this->values.end()) {
        //    return false;
        //} else {
        //    this->SetValue(*iter);
        //    return true;
        //}
    } catch (...) {}
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
        if (setDirty)
            this->setDirty();
    }
}


/*
 * FlexEnumParam::ValueString
 */
std::string FlexEnumParam::ValueString(void) const {
    return val;
}
