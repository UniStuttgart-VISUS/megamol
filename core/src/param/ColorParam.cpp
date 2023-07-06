/**
 * MegaMol
 * Copyright (c) 2008, MegaMol Dev Team
 * All rights reserved.
 */

#include "mmcore/param/ColorParam.h"

#include "mmcore/utility/ColourParser.h"

using namespace megamol::core::param;

ColorParam::ColorParam(const ColorType& initVal) : AbstractParam(), val() {
    this->InitPresentation(AbstractParamPresentation::ParamType::COLOR);
    this->SetValue(initVal);
}

ColorParam::ColorParam(float initR, float initG, float initB, float initA) : AbstractParam(), val() {
    this->InitPresentation(AbstractParamPresentation::ParamType::COLOR);
    this->SetValue({initR, initG, initB, initA});
}

ColorParam::ColorParam(const std::string& initVal) : AbstractParam(), val() {
    this->InitPresentation(AbstractParamPresentation::ParamType::COLOR);
    this->ParseValue(initVal);
}

bool ColorParam::ParseValue(std::string const& v) {

    // Checked color syntax:
    // 1] #123 #1234 #123456 #12345678
    // 2] Colour(1.0;0.5;1.0;1.0) Colour(1.0;0.5;1.0)
    // 3] 'Named Colour', e.g. Red
    try {
        float vParsed[4];
        if (core::utility::ColourParser::FromString(v.c_str(), 4, vParsed)) {
            ColorType vConverted = {vParsed[0], vParsed[1], vParsed[2], vParsed[3]};
            this->SetValue(vConverted);
            return true;
        }
    } catch (...) {}

    return false;
}

std::string ColorParam::ValueString() const {
    return core::utility::ColourParser::ToString(this->val[0], this->val[1], this->val[2], this->val[3]).PeekBuffer();
}

void ColorParam::SetValue(const ColorType& v, bool setDirty) {
    if (v != this->val) {
        this->val = v;
        this->indicateParamChange();
        if (setDirty)
            this->setDirty();
    }
}
