/**
 * MegaMol
 * Copyright (c) 2021, MegaMol Dev Team
 * All rights reserved.
 */

#pragma once

#include <sstream>

#include "GenericParam.h"

namespace megamol::core::param {

class FloatParam : public GenericParam<float, AbstractParamPresentation::ParamType::FLOAT> {
public:
    FloatParam(float initVal) : Super(initVal) {}

    FloatParam(float initVal, float minVal) : Super(initVal, minVal) {}

    FloatParam(float initVal, float minVal, float maxVal) : Super(initVal, minVal, maxVal) {}

    FloatParam(float initVal, float minVal, float maxVal, float stepSize) : Super(initVal, minVal, maxVal, stepSize) {}

    ~FloatParam() override = default;

    bool ParseValue(std::string const& v) override {
        try {
            this->SetValue(std::stof(v));
            return true;
        } catch (...) {}
        return false;
    }

    std::string ValueString() const override {
        return std::to_string(Value());
    }

private:
    using Super = GenericParam<float, AbstractParamPresentation::ParamType::FLOAT>;
};

} // namespace megamol::core::param
