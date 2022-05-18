/*
 * BoolParam.h
 *
 * Copyright (C) 2021 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#pragma once

#include "GenericParam.h"


namespace megamol::core::param {

class BoolParam : public GenericParam<bool, AbstractParamPresentation::ParamType::BOOL> {
public:
    BoolParam(float initVal) : Super(initVal) {}

    virtual ~BoolParam() = default;

    std::string Definition() const override {
        return "MMBOOL";
    }

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
