/**
 * MegaMol
 * Copyright (c) 2008, MegaMol Dev Team
 * All rights reserved.
 */

#include "mmcore/param/TernaryParam.h"

using namespace megamol::core::param;


/*
 * TernaryParam::TernaryParam
 */
TernaryParam::TernaryParam(const vislib::math::Ternary& initVal) : AbstractParam(), val() {
    this->InitPresentation(AbstractParamPresentation::ParamType::TERNARY);
    this->SetValue(initVal);
}


/*
 * TernaryParam::~TernaryParam
 */
TernaryParam::~TernaryParam(void) {
    // intentionally empty
}


/*
 * TernaryParam::ParseValue
 */
bool TernaryParam::ParseValue(std::string const& v) {
    return this->val.Parse(v.c_str());
}


/*
 * TernaryParam::SetValue
 */
void TernaryParam::SetValue(vislib::math::Ternary v, bool setDirty) {
    if (this->val != v) {
        this->val = v;
        this->indicateChange();
        if (setDirty)
            this->setDirty();
    }
}


/*
 * TernaryParam::ValueString
 */
std::string TernaryParam::ValueString(void) const {
    return this->val.ToStringA().PeekBuffer();
}
