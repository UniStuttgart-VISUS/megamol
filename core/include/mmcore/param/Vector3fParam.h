/*
 * Vector3fParam.h
 *
 * Copyright (C) 2021 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#pragma once

#include <array>
#include <sstream>

#include "GenericParam.h"
#include "vislib/Array.h"
#include "vislib/StringTokeniser.h"
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

    virtual ~Vector3fParam() = default;

    std::array<float, 3> getArray() const {
        return std::array<float, 3>({this->Value().GetX(), this->Value().GetY(), this->Value().GetZ()});
    }

    std::string Definition() const override {
        std::string name = "MMVC3F";
        std::string return_str;
        return_str.resize(6 + 6 * sizeof(float));
        std::copy(name.begin(), name.end(), return_str.begin());
        for (int i = 0; i < 3; ++i) {
            std::copy(reinterpret_cast<char const*>(&MinValue()[i]),
                reinterpret_cast<char const*>(&MinValue()[i]) + sizeof(float),
                return_str.begin() + name.size() + i * sizeof(float));
        }
        for (int i = 0; i < 3; ++i) {
            std::copy(reinterpret_cast<char const*>(&MaxValue()[i]),
                reinterpret_cast<char const*>(&MaxValue()[i]) + sizeof(float),
                return_str.begin() + name.size() + 3 * sizeof(float) + i * sizeof(float));
        }
        return return_str;
    }

    bool ParseValue(std::string const& v) override {
        vislib::Array<vislib::TString> comps = vislib::TStringTokeniser::Split(v.c_str(), _T(";"), true);
        if (comps.Count() == 3) {
            try {
                comps[0].TrimSpaces();
                comps[1].TrimSpaces();
                comps[2].TrimSpaces();
                float x = static_cast<float>(vislib::TCharTraits::ParseDouble(comps[0]));
                float y = static_cast<float>(vislib::TCharTraits::ParseDouble(comps[1]));
                float z = static_cast<float>(vislib::TCharTraits::ParseDouble(comps[2]));

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
