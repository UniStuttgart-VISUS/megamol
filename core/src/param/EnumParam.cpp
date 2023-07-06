/**
 * MegaMol
 * Copyright (c) 2008, MegaMol Dev Team
 * All rights reserved.
 */

#include "mmcore/param/EnumParam.h"

#include "vislib/IllegalStateException.h"

using namespace megamol::core::param;


/*
 * EnumParam::EnumParam
 */
EnumParam::EnumParam(int initVal) : AbstractParam(), val(), typepairs() {
    this->InitPresentation(AbstractParamPresentation::ParamType::ENUM);
    this->SetValue(initVal);
}


/*
 * EnumParam::~EnumParam
 */
EnumParam::~EnumParam() {
    // intentionally empty
}


/*
 * EnumParam::ClearTypePairs
 */
void megamol::core::param::EnumParam::ClearTypePairs() {
    if (this->isSlotPublic()) {
        throw vislib::IllegalStateException(
            "You must not modify an enum parameter which is already public", __FILE__, __LINE__);
    }
    this->typepairs.clear();
}


/*
 * EnumParam::ParseValue
 */
bool EnumParam::ParseValue(std::string const& v) {
    for (auto const& el : typepairs) {
        if (el.second == v) {
            this->SetValue(el.first);
            return true;
        }
    }
    try {
        this->SetValue(std::stoi(v));
        return true;
    } catch (...) {}
    return false;
}


/*
 * EnumParam::SetTypePair
 */
EnumParam* EnumParam::SetTypePair(int value, const char* name) {
    if (this->isSlotPublic()) {
        throw vislib::IllegalStateException(
            "You must not modify an enum parameter which is already public", __FILE__, __LINE__);
    }
    this->typepairs[value] = name;
    return this;
}


/*
 * EnumParam::SetValue
 */
void EnumParam::SetValue(int v, bool setDirty) {
    if (this->val != v) {
        this->val = v;
        this->indicateParamChange();
        if (setDirty)
            this->setDirty();
    }
}


/*
 * EnumParam::ValueString
 */
std::string EnumParam::ValueString() const {
    if (typepairs.count(val) > 0) {
        return typepairs.at(val);
    }
    return std::to_string(val);
}
