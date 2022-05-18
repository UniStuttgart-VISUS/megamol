/*
 * IntParam.h
 *
 * Copyright (C) 2021 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#pragma once

#include <sstream>

#include "GenericParam.h"

namespace megamol::core::param {

class IntParam : public GenericParam<int, AbstractParamPresentation::ParamType::INT> {
public:
    IntParam(int initVal) : Super(initVal) {}

    IntParam(int initVal, int minVal) : Super(initVal, minVal) {}

    IntParam(int initVal, int minVal, int maxVal) : Super(initVal, minVal, maxVal) {}

    IntParam(int initVal, int minVal, int maxVal, int stepSize) : Super(initVal, minVal, maxVal, stepSize) {}

    virtual ~IntParam() = default;

    std::string Definition() const override {
        std::ostringstream outDef;
        outDef << "MMINTR";
        outDef.write(reinterpret_cast<char const*>(&MinValue()), sizeof(MinValue()));
        outDef.write(reinterpret_cast<char const*>(&MaxValue()), sizeof(MaxValue()));
        outDef.write(reinterpret_cast<char const*>(&StepSize()), sizeof(StepSize()));

        return outDef.str();
    }

    bool ParseValue(std::string const& v) override {
        try {
            this->SetValue(vislib::TCharTraits::ParseInt(v.c_str()));
            return true;
        } catch (...) {}
        return false;
    }

    std::string ValueString() const override {
        return std::to_string(Value());
    }

private:
    using Super = GenericParam<int, AbstractParamPresentation::ParamType::INT>;
};

} // namespace megamol::core::param
