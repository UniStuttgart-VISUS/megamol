/*
 * StringParam.h
 *
 * Copyright (C) 2021 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#pragma once

#include <string>

#include "GenericParam.h"


namespace megamol::core::param {

class StringParam : public GenericParam<std::string, AbstractParamPresentation::ParamType::STRING> {
public:
    StringParam(std::string const& initVal) : Super(initVal) {}

    virtual ~StringParam() = default;

    std::string Definition() const override {
        return "MMSTRA";
    }

    bool ParseValue(std::string const& v) override {
        try {
            this->SetValue(v);
            return true;
        } catch (...) {}
        return false;
    }

    std::string ValueString() const override {
        return Value();
    }

private:
    using Super = GenericParam<std::string, AbstractParamPresentation::ParamType::STRING>;
};

} // namespace megamol::core::param
