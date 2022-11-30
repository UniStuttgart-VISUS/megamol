/**
 * MegaMol
 * Copyright (c) 2008, MegaMol Dev Team
 * All rights reserved.
 */

#pragma once

#include "AbstractParam.h"
#include "vislib/math/Ternary.h"

namespace megamol::core::param {

/**
 * class for ternary parameter objects
 */
class TernaryParam : public AbstractParam {
public:
    /**
     * Ctor.
     *
     * @param initVal The initial state value
     */
    TernaryParam(const vislib::math::Ternary& initVal);

    /**
     * Dtor.
     */
    virtual ~TernaryParam();

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
     * Sets the value of the parameter and optionally sets the dirty flag
     * of the owning parameter slot.
     *
     * @param v the new value for the parameter
     * @param setDirty If 'true' the dirty flag of the owning parameter
     *                 slot is set and the update callback might be called.
     */
    void SetValue(vislib::math::Ternary v, bool setDirty = true);

    /**
     * Gets the value of the parameter
     *
     * @return The value of the parameter
     */
    inline const vislib::math::Ternary& Value() const {
        return this->val;
    }

    /**
     * Returns the value of the parameter as string.
     *
     * @return The value of the parameter as string.
     */
    std::string ValueString() const override;

    /**
     * Gets the value of the parameter
     *
     * @return The value of the parameter
     */
    inline operator vislib::math::Ternary() const {
        return this->val;
    }

private:
    /** The value of the parameter */
    vislib::math::Ternary val;
};


} // namespace megamol::core::param
