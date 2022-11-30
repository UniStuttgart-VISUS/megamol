/**
 * MegaMol
 * Copyright (c) 2008, MegaMol Dev Team
 * All rights reserved.
 */

#pragma once

#include <map>
#include <string>

#include "AbstractParam.h"

namespace megamol::core::param {

/**
 * class for enumeration parameter objects
 */
class EnumParam : public AbstractParam {
public:
    /**
     * Ctor.
     *
     * @param initVal The initial value
     */
    EnumParam(int initVal);

    /**
     * Dtor.
     */
    ~EnumParam() override;

    /**
     * Clears TypePairs storage.
     */
    virtual void ClearTypePairs();

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
    EnumParam* SetTypePair(int value, const char* name);

    /**
     * Sets the value of the parameter and optionally sets the dirty flag
     * of the owning parameter slot.
     *
     * @param v the new value for the parameter
     * @param setDirty If 'true' the dirty flag of the owning parameter
     *                 slot is set and the update callback might be called.
     */
    void SetValue(int v, bool setDirty = true);

    /**
     * Gets the value of the parameter
     *
     * @return The value of the parameter
     */
    inline int Value() const {
        return this->val;
    }

    /**
     * Returns the TypePairs storage.
     *
     * @return The TypePairs storage.
     */
    inline std::map<int, std::string> getMap() {
        return this->typepairs;
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
    inline operator int() const {
        return this->val;
    }

    /**
     * Answers the number of currently owned typepairs
     *
     * @return The number of currently owned typepairs
     */
    inline std::size_t ContentCount() const {
        return typepairs.size();
    }

private:
    /** The value of the parameter */
    int val;

    /** The type pairs for values and names */
    std::map<int, std::string> typepairs;
};


} // namespace megamol::core::param
