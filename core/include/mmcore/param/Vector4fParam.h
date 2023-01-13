/**
 * MegaMol
 * Copyright (c) 2021, MegaMol Dev Team
 * All rights reserved.
 */

#pragma once

#include <sstream>

#include "GenericParam.h"
#include "mmcore/utility/String.h"
#include "vislib/math/Vector.h"

namespace megamol::core::param {

class Vector4fParam
        : public GenericParam<vislib::math::Vector<float, 4>, AbstractParamPresentation::ParamType::VECTOR4F> {
public:
    Vector4fParam(vislib::math::Vector<float, 4> const& initVal) : Super(initVal) {}

    Vector4fParam(vislib::math::Vector<float, 4> const& initVal, vislib::math::Vector<float, 4> const& minVal)
            : Super(initVal, minVal) {}

    Vector4fParam(vislib::math::Vector<float, 4> const& initVal, vislib::math::Vector<float, 4> const& minVal,
        vislib::math::Vector<float, 4> const& maxVal)
            : Super(initVal, minVal, maxVal) {}

    ~Vector4fParam() override = default;

    bool ParseValue(std::string const& v) override {
        const auto& segments = utility::string::Split(v, ';');
        if (segments.size() == 4) {
            try {
                float x = std::stof(utility::string::Trim(segments[0]));
                float y = std::stof(utility::string::Trim(segments[1]));
                float z = std::stof(utility::string::Trim(segments[2]));
                float w = std::stof(utility::string::Trim(segments[3]));
                this->SetValue(vislib::math::Vector<float, 4>(x, y, z, w));
                return true;
            } catch (...) {}
        }
        return false;
    }

    std::string ValueString() const override {
        std::stringstream stream;
        stream.precision(std::numeric_limits<float>::max_digits10);
        stream << Value()[0] << ";" << Value()[1] << ";" << Value()[2] << ";" << Value()[3];
        return stream.str();
    }

private:
    using Super = GenericParam<vislib::math::Vector<float, 4>, AbstractParamPresentation::ParamType::VECTOR4F>;
};

} // namespace megamol::core::param
