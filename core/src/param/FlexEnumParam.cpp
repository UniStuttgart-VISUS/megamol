/**
 * MegaMol
 * Copyright (c) 2017, MegaMol Dev Team
 * All rights reserved.
 */

#include "mmcore/param/FlexEnumParam.h"

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
FlexEnumParam::~FlexEnumParam() {
    // intentionally empty
}


/*
 * FlexEnumParam::ClearTypePairs
 */
void megamol::core::param::FlexEnumParam::ClearValues() {
    this->SetHash(this->GetHash() + 1);
    this->values.clear();

    this->indicateParamChange();
    this->indicatePresentationChange();
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
        
        this->indicateParamChange();
        this->indicatePresentationChange();
    }
    return this;
}


/*
 * FlexEnumParam::SetValue
 */
void FlexEnumParam::SetValue(const std::string& v, bool setDirty) {
    if (this->val != v) {
        this->val = v;
        this->indicateParamChange();
        if (setDirty)
            this->setDirty();
    }
}


/*
 * FlexEnumParam::ValueString
 */
std::string FlexEnumParam::ValueString() const {
    return val;
}
