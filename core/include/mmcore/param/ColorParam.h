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
 * class for transferfunction (1D) parameter objects
 */
class MEGAMOLCORE_API ColorParam : public AbstractParam {
public:
    /**
     * Ctor.
     *
     * @param initVal The initial value
     * @param visible If 'true' the parameter is visible in the gui.
     */
    ColorParam(const vislib::StringA& initVal);

    /**
     * Ctor.
     *
     * @param initVal The initial value
     * @param visible If 'true' the parameter is visible in the gui.
     */
    ColorParam(const vislib::StringW& initVal);

    /**
     * Ctor.
     *
     * @param initVal The initial value
     */
    ColorParam(const char *initVal);

    /**
     * Ctor.
     *
     * @param initVal The initial value
     */
    ColorParam(const wchar_t *initVal);

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
    * Sets the value of the parameter and optionally sets the dirty flag
    * of the owning parameter slot.
    *
    * @param v the new value for the parameter
    * @param setDirty If 'true' the dirty flag of the owning parameter
    *                 slot is set and the update callback might be called.
    */
    void SetValue(const vislib::StringA& v, bool setDirty = true);

    /**
    * Sets the value of the parameter and optionally sets the dirty flag
    * of the owning parameter slot.
    *
    * @param v the new value for the parameter
    * @param setDirty If 'true' the dirty flag of the owning parameter
    *                 slot is set and the update callback might be called.
    */
    void SetValue(const vislib::StringW& v, bool setDirty = true);

    /**
    * Sets the value of the parameter and optionally sets the dirty flag
    * of the owning parameter slot.
    *
    * @param v the new value for the parameter
    * @param setDirty If 'true' the dirty flag of the owning parameter
    *                 slot is set and the update callback might be called.
    */
    void SetValue(const char *v, bool setDirty = true);

    /**
    * Sets the value of the parameter and optionally sets the dirty flag
    * of the owning parameter slot.
    *
    * @param v the new value for the parameter
    * @param setDirty If 'true' the dirty flag of the owning parameter
    *                 slot is set and the update callback might be called.
    */
    void SetValue(const wchar_t *v, bool setDirty = true);

    /**
    * Gets the value of the parameter
    *
    * @return The value of the parameter
    */
    inline const vislib::TString& Value(void) const {
        return this->val;
    }

    /**
    * Returns the value of the parameter as string.
    *
    * @return The value of the parameter as string.
    */
    virtual vislib::TString ValueString(void) const override;

    /**
    * Gets the value of the parameter
    *
    * @return The value of the parameter
    */
    inline operator const vislib::TString&(void) const {
        return this->val;
    }
private:
    /** The value of the parameter */
    vislib::TString val;
}; /* end class ColorParam */

} /* end namespace param */
} /* end namespace core */
} /* end namespace megamol */

#endif /* end ifndef MEGAMOLCORE_COLORPARAM_H_INCLUDED */