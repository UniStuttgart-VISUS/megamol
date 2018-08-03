/*
 * ColorParam.h
 *
 * Copyright (C) 2018 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_COLORPARAM_H_INCLUDED
#define MEGAMOLCORE_COLORPARAM_H_INCLUDED

#include "mmcore/api/MegaMolCore.std.h"
#include "AbstractParam.h"

namespace megamol {
namespace core {
namespace param {


/**
 * Class for 32bit RGBA color parameters with each channel between 0.0 and 1.0.
 */
class MEGAMOLCORE_API ColorParam : public AbstractParam {
public:
    typedef float Type[4];

    /**
     * Ctor.
     *
     * @param initVal The initial value
     */
    ColorParam(const Type& initVal);

    /**
     * Ctor.
     *
     * @param initVal The initial value
     */
    ColorParam(const vislib::TString& initVal);

    /**
     * Dtor.
     */
    virtual ~ColorParam(void);

    /**
     * Returns a machine-readable definition of the parameter.
     *
     * @param outDef A memory block to receive a machine-readable
     *               definition of the parameter.
     */
    virtual void Definition(vislib::RawStorage& outDef) const override;

    /**
     * Tries to parse the given string as value for this parameter and
     * sets the new value if successful. This also triggers the update
     * mechanism of the slot this parameter is assigned to.
     *
     * @param v The new value for the parameter as string.
     *
     * @return 'true' on success, 'false' otherwise.
     */
    virtual bool ParseValue(const vislib::TString& v) override;

    /**
    * Returns the value of the parameter as string.
    *
    * @return The value of the parameter as string.
    */
    virtual vislib::TString ValueString(void) const override;

    /**
    * Sets the value of the parameter and optionally sets the dirty flag
    * of the owning parameter slot.
    *
    * @param v the new value for the parameter
    * @param setDirty If 'true' the dirty flag of the owning parameter
    *                 slot is set and the update callback might be called.
    */
    void SetValue(const Type& v, bool setDirty = true);

    /**
    * Gets the value of the parameter
    *
    * @return The value of the parameter
    */
    inline const Type& Value(void) const {
        return this->val;
    }

    /**
     * Returns a 32bit RGBA color.
     */
    inline operator const float*(void) const { 
        return this->val;
    }

private:
    /** The value of the parameter */
    Type val;
}; /* end class ColorParam */

} /* end namespace param */
} /* end namespace core */
} /* end namespace megamol */

#endif /* end ifndef MEGAMOLCORE_COLORPARAM_H_INCLUDED */