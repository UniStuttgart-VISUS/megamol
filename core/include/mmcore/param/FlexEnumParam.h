/**
 * MegaMol
 * Copyright (c) 2017, MegaMol Dev Team
 * All rights reserved.
 */

#pragma once

#include <set>
#include <string>

#include "AbstractParam.h"

namespace megamol::core::param {

/**
 * class for enumeration parameter objects that can change
 * valid values at runtime. Handle with care since this param
 * cannot check values for validity as long as project file loading
 * sets values sooner than the user code defining the valid
 * contents is executed.
 */
class FlexEnumParam : public AbstractParam {
public:
    typedef std::set<std::string> Storage_t;

    /**
     * Ctor.
     *
     * @param initVal The initial value
     */
    FlexEnumParam(const std::string& initVal);

    /**
     * Dtor.
     */
    ~FlexEnumParam() override;

    /**
     * Clears TypePairs storage.
     */
    virtual void ClearValues();

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
     * Sets a type pair for the enum type. Although the parameter can hold
     * any integer number as value, gui mechanisms will only be able to
     * represent values associated with a name.
     *
     * Calling the method for a value which already has a name will
     * overwrite the previously set name.
     *
     * You must not call this method after the slot this parameter is
     * assigned to has been made public.
     *
     * @param value The value to set the name for.
     * @param name The name of the value specified.
     *
     * @return 'this'
     */
    FlexEnumParam* AddValue(const std::string& name);

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
     * Gets the value of the parameter
     *
     * @return The value of the parameter
     */
    inline const std::string& Value() const {
        return this->val;
    }

    /**
     * Returns the values storage.
     *
     * @return The values storage.
     */
    inline Storage_t getStorage() {
        return this->values;
    }

    /**
     * Returns the value of the parameter as string.
     *
     * @return The value of the parameter as string.
     */
    std::string ValueString() const override;

    /**
     * Answers the number of currently owned typepairs
     *
     * @return The number of currently owned typepairs
     */
    inline size_t ContentCount() const {
        return values.size();
    }

private:
    /** The value of the parameter */
    std::string val;

    /** The type pairs for values and names */
    Storage_t values;
};


} // namespace megamol::core::param
