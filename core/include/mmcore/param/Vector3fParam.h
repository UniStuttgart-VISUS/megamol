/**
 * MegaMol
 * Copyright (c) 2021, MegaMol Dev Team
 * All rights reserved.
 */

#pragma once

#include <array>
#include <sstream>

#include "GenericParam.h"
#include "mmcore/utility/String.h"
#include "vislib/math/Vector.h"

namespace megamol::core::param {

class Vector3fParam
        : public GenericParam<vislib::math::Vector<float, 3>, AbstractParamPresentation::ParamType::VECTOR3F> {
public:
    Vector3fParam(vislib::math::Vector<float, 3> const& initVal) : Super(initVal) {}

    Vector3fParam(vislib::math::Vector<float, 3> const& initVal, vislib::math::Vector<float, 3> const& minVal)
            : Super(initVal, minVal) {}

    Vector3fParam(vislib::math::Vector<float, 3> const& initVal, vislib::math::Vector<float, 3> const& minVal,
        vislib::math::Vector<float, 3> const& maxVal)
            : Super(initVal, minVal, maxVal) {}

    ~Vector3fParam() override = default;

    std::array<float, 3> getArray() const {
        return std::array<float, 3>({this->Value().GetX(), this->Value().GetY(), this->Value().GetZ()});
    }

    bool ParseValue(std::string const& v) override {
        const auto& segments = utility::string::Split(v, ';');
        if (segments.size() == 3) {
            try {
                float x = std::stof(utility::string::TrimCopy(segments[0]));
                float y = std::stof(utility::string::TrimCopy(segments[1]));
                float z = std::stof(utility::string::TrimCopy(segments[2]));
                this->SetValue(vislib::math::Vector<float, 3>(x, y, z));
                return true;
            } catch (...) {}
        }
        return false;
    }

    std::string ValueString() const override {
        std::stringstream stream;
        stream.precision(std::numeric_limits<float>::max_digits10);
        stream << Value()[0] << ";" << Value()[1] << ";" << Value()[2];
        return stream.str();
    }

private:
    using Super = GenericParam<vislib::math::Vector<float, 3>, AbstractParamPresentation::ParamType::VECTOR3F>;
};

} // namespace megamol::core::param
