/**
 * MegaMol
 * Copyright (c) 2021, MegaMol Dev Team
 * All rights reserved.
 */

#pragma once

#include "GenericParam.h"

namespace megamol::core::param {

class BoolParam : public GenericParam<bool, AbstractParamPresentation::ParamType::BOOL> {
public:
    BoolParam(bool initVal) : Super(initVal) {}

    ~BoolParam() override = default;

    bool ParseValue(std::string const& v) override {
        try {
            this->SetValue(vislib::TCharTraits::ParseBool(v.c_str()));
            return true;
        } catch (...) {}
        return false;
    }

    std::string ValueString() const override {
        return Value() ? "true" : "false";
    }

private:
    using Super = GenericParam<bool, AbstractParamPresentation::ParamType::BOOL>;
};

} // namespace megamol::core::param
