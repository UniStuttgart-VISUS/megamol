/*
 * EnumParam.h
 *
 * Copyright (C) 2008 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_ENUMPARAM_H_INCLUDED
#define MEGAMOLCORE_ENUMPARAM_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "AbstractParam.h"
#include "vislib/Map.h"
#include "vislib/String.h"
#include "vislib/tchar.h"


namespace megamol {
namespace core {
namespace param {


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
    virtual ~EnumParam(void);

    /**
     * Clears TypePairs storage.
     */
    virtual void ClearTypePairs(void);

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
    EnumParam* SetTypePair(int value, const wchar_t* name);

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
    inline int Value(void) const {
        return this->val;
    }

    /**
     * Returns the TypePairs storage.
     *
     * @return The TypePairs storage.
     */
    inline vislib::Map<int, vislib::TString> getMap() {
        return this->typepairs;
    }

    /**
     * Returns the value of the parameter as string.
     *
     * @return The value of the parameter as string.
     */
    std::string ValueString(void) const override;

    /**
     * Gets the value of the parameter
     *
     * @return The value of the parameter
     */
    inline operator int(void) const {
        return this->val;
    }

    /**
     * Answers the number of currently owned typepairs
     *
     * @return The number of currently owned typepairs
     */
    inline size_t ContentCount() const {
        return typepairs.Count();
    }

private:
    /** The value of the parameter */
    int val;

#ifdef _WIN32
#pragma warning(disable : 4251)
#endif /* _WIN32 */
    /** The type pairs for values and names */
    vislib::Map<int, vislib::TString> typepairs;
#ifdef _WIN32
#pragma warning(default : 4251)
#endif /* _WIN32 */
};


} /* end namespace param */
} /* end namespace core */
} /* end namespace megamol */

#endif /* MEGAMOLCORE_ENUMPARAM_H_INCLUDED */
