/**
 * MegaMol
 * Copyright (c) 2021, MegaMol Dev Team
 * All rights reserved.
 */

#pragma once

#include <sstream>

#include "GenericParam.h"
#include "vislib/Array.h"
#include "vislib/StringTokeniser.h"
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
        vislib::Array<vislib::TString> comps = vislib::TStringTokeniser::Split(v.c_str(), _T(";"), true);
        if (comps.Count() == 4) {
            try {
                comps[0].TrimSpaces();
                comps[1].TrimSpaces();
                comps[2].TrimSpaces();
                comps[3].TrimSpaces();
                float x = static_cast<float>(vislib::TCharTraits::ParseDouble(comps[0]));
                float y = static_cast<float>(vislib::TCharTraits::ParseDouble(comps[1]));
                float z = static_cast<float>(vislib::TCharTraits::ParseDouble(comps[2]));
                float w = static_cast<float>(vislib::TCharTraits::ParseDouble(comps[3]));

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
