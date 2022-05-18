/*
 * ColorParam.h
 *
 * Copyright (C) 2018 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_COLORPARAM_H_INCLUDED
#define MEGAMOLCORE_COLORPARAM_H_INCLUDED

#include "AbstractParam.h"

#include <array>


namespace megamol {
namespace core {
namespace param {


/**
 * Class for 32bit RGBA color parameters with each channel between 0.0 and 1.0.
 */
class ColorParam : public AbstractParam {
public:
    typedef std::array<float, 4> ColorType;

    /**
     * Ctor.
     *
     * @param initVal The initial value
     */
    ColorParam(const ColorType& initVal);

    /**
     * Ctor.
     *
     * @param initVal The initial value
     */
    ColorParam(float initR, float initG, float initB, float initA);

    /**
     * Ctor.
     *
     * @param initVal The initial value
     */
    ColorParam(const vislib::TString& initVal);

    /**
     * Dtor.
     */
    virtual ~ColorParam(void) = default;

    /**
     * Returns a machine-readable definition of the parameter.
     *
     * @param outDef A memory block to receive a machine-readable
     *               definition of the parameter.
     */
    std::string Definition() const override;

    /**
     * Tries to parse the given string as value for this parameter and
     * sets the new value if successful. This also triggers the update
     * mechanism of the slot this parameter is assigned to.
     *
     * @param v The new value for the parameter as string.
     *
     * @return 'true' on success, 'false' otherwise.
     */
    bool ParseValue(std::string const& v) override;

    /**
     * Returns the value of the parameter as string.
     *
     * @return The value of the parameter as string.
     */
    std::string ValueString(void) const override;

    /**
     * Sets the value of the parameter and optionally sets the dirty flag
     * of the owning parameter slot.
     *
     * @param v the new value for the parameter
     * @param setDirty If 'true' the dirty flag of the owning parameter
     *                 slot is set and the update callback might be called.
     */
    void SetValue(const ColorType& v, bool setDirty = true);

    /**
     * Gets the value of the parameter
     *
     * @return The value of the parameter
     */
    inline const ColorType& Value(void) const {
        return this->val;
    }

    /**
     * Gets the value of the parameter
     *
     * @return The value of the parameter
     */
    inline const void Value(float& outR, float& outG, float& outB, float& outA) const {
        outR = this->val[0];
        outG = this->val[1];
        outB = this->val[2];
        outA = this->val[3];
    }

    /**
     * Gets the color array
     *
     * @return The array of values
     */
    inline const std::array<float, 4> GetArray() const {
        return val;
    }

    /**
     * Gets the value of the parameter
     *
     * @return The value of the parameter
     */
    inline const void Value(float& outR, float& outG, float& outB) const {
        float a;
        this->Value(outR, outG, outB, a);
    }

    /**
     * Returns a 32bit RGBA color.
     */
    inline operator const ColorType(void) const {
        return this->val;
    }

private:
    /** The value of the parameter */
    ColorType val;

}; /* end class ColorParam */

} /* end namespace param */
} /* end namespace core */
} /* end namespace megamol */

#endif /* end ifndef MEGAMOLCORE_COLORPARAM_H_INCLUDED */
