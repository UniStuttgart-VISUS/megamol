/*
 * TransferFunctionParam.h
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_TRANSFERFUNCTIONPARAM_H_INCLUDED
#define MEGAMOLCORE_TRANSFERFUNCTIONPARAM_H_INCLUDED


#include <string>

#include "mmcore/api/MegaMolCore.std.h"
#include "AbstractParam.h"

#include "vislib/String.h"
		

namespace megamol {
namespace core {
namespace param {


/**
 * Class for parameter holding transfer function as JSON string.
 */
class MEGAMOLCORE_API TransferFunctionParam : public AbstractParam {
public:

    /**
     * Ctor.
     *
     * @param initVal The initial value
     */
    TransferFunctionParam(const std::string& initVal);

    /**
     * Ctor.
     *
     * @param initVal The initial value
     */
    TransferFunctionParam(const char *initVal);

    /**
     * Ctor.
     *
     * @param initVal The initial value
     * @param visible If 'true' the parameter is visible in the gui.
     */
    TransferFunctionParam(const vislib::StringA& initVal);

    /**
     * Dtor.
     */
    virtual ~TransferFunctionParam(void);

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
    void SetValue(const std::string& v, bool setDirty = true);

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
    inline const std::string& Value(void) const {
        return this->val;
    }

    /**
     * Gets the value of the parameter
     *
     * @return The value of the parameter
     */
    inline operator const std::string&(void) const {
        return this->val;
    }

private:

    /** The value of the parameter */
    std::string val;

}; /* end class TransferFunctionParam */

} /* end namespace param */
} /* end namespace core */
} /* end namespace megamol */

#endif /* end ifndef MEGAMOLCORE_TRANSFERFUNCTIONPARAM_H_INCLUDED */