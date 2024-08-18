#pragma once

#include "CUDAUtils.h"

namespace megamol {
namespace optix_owl {
template<typename FType, typename BaseType, BaseType D>
class FixedPoint {
    static_assert(std::is_floating_point_v<FType>, "FType must be floating point type");
    static_assert(std::is_integral_v<BaseType>, "BaseType must be integral type");

public:
    constexpr static BaseType factor = 1 << D;
    constexpr static BaseType half_factor = 1 << (D - 1);

    CU_CALLABLE FixedPoint() = default;

    CU_CALLABLE explicit FixedPoint(FType val) {
        *this = val;
    }

    CU_CALLABLE explicit FixedPoint(BaseType val) {
        *this = val;
    }

    CU_CALLABLE FixedPoint(FixedPoint const& rhs) = default;
    CU_CALLABLE FixedPoint& operator=(FixedPoint const& rhs) = default;
    CU_CALLABLE FixedPoint(FixedPoint&& rhs) = default;
    CU_CALLABLE FixedPoint& operator=(FixedPoint&& rhs) = default;

    /*FixedPoint(FixedPoint const& rhs) {
        *this = rhs;
    }

    FixedPoint& operator=(FixedPoint const& rhs) {
        value_ = rhs.value_;
    }

    FixedPoint(FixedPoint&& rhs) {
        *this = std::move(rhs);
    }

    FixedPoint& operator=(FixedPoint&& rhs) {
        value_ = std::exchange(rhs.value_, static_cast<BaseType>(0));
        return *this;
    }*/

    CU_CALLABLE FixedPoint& operator=(FType val) {
        value_ = static_cast<BaseType>(val * factor);
        return *this;
    }

    CU_CALLABLE FixedPoint& operator=(BaseType val) {
        value_ = val;
        return *this;
    }

    CU_CALLABLE FixedPoint operator+(FixedPoint const& rhs) const {
        FixedPoint ret;
        ret.value_ = value_ + rhs.value_;
        return ret;
    }

    CU_CALLABLE FixedPoint operator-(FixedPoint const& rhs) const {
        FixedPoint ret;
        ret.value_ = value_ - rhs.value_;
        return ret;
    }

    CU_CALLABLE FixedPoint operator*(FixedPoint const& rhs) const {
        FixedPoint ret;
        ret.value_ = ((value_ * rhs.value_) + half_factor) >> D;
        return ret;
    }

    CU_CALLABLE FixedPoint operator/(FixedPoint const& rhs) const {
        FixedPoint ret;
        ret.value_ = (value_ << D) / rhs.value_;
        return ret;
    }

    CU_CALLABLE operator FType() const {
        return static_cast<FType>(value_) / static_cast<FType>(factor);
    }

    CU_CALLABLE operator BaseType() const {
        return value_;
    }

    CU_CALLABLE operator BaseType&() {
        return value_;
    }

private:
    BaseType value_;
};
} // namespace optix_owl
} // namespace megamol
