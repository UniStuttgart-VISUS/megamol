/*
 * AbstractParam.h
 *
 * Copyright (C) 2008 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_ABSTRACTPARAM_H_INCLUDED
#define MEGAMOLCORE_ABSTRACTPARAM_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "mmcore/param/AbstractParamPresentation.h"

#include "vislib/RawStorage.h"
#include "vislib/String.h"
#include "vislib/tchar.h"

#include <functional>


namespace megamol {
namespace core {
namespace param {


/** forward declaration of owning class */
class AbstractParamSlot;


/**
 * Abstract base class for all parameter objects
 */
class AbstractParam : public AbstractParamPresentation {
public:
    friend class AbstractParamSlot;

    using ParamChangeCallback = std::function<void(AbstractParamSlot*)>;

    /**
     * Dtor.
     */
    virtual ~AbstractParam(void);

    /**
     * Returns a machine-readable definition of the parameter.
     *
     * @param outDef A memory block to receive a machine-readable
     *               definition of the parameter.
     */
    virtual std::string Definition() const = 0;

    /**
     * Tries to parse the given string as value for this parameter and
     * sets the new value if successful. This also triggers the update
     * mechanism of the slot this parameter is assigned to.
     *
     * @param v The new value for the parameter as string.
     *
     * @return 'true' on success, 'false' otherwise.
     */
    virtual bool ParseValue(std::string const& v) = 0;

    /**
     * Returns the value of the parameter as string.
     *
     * @return The value of the parameter as string.
     */
    virtual std::string ValueString(void) const = 0;

    /**
     * Must be public for Button Press - Manuel Graeber
     * Sets the dirty flag of the owning parameter slot and might call the
     * update callback.
     */
    void setDirty(void);

    /**
     * Returns the value of the hash.
     *
     * @return The value of the hash.
     */
    inline uint64_t GetHash(void) const {
        return this->hash;
    }

    /**
     * Sets the value of the hash.
     *
     * @param hash The value of the hash.
     */
    inline void SetHash(const uint64_t& hash) {
        this->hash = hash;
    }

    /**
     * Returns the has_changed flag and resets the flag to false.
     *
     * @return has_changed
     */
    bool ConsumeHasChanged() {
        auto val = has_changed;
        has_changed = false;
        return val;
    }

    void setChangeCallback(ParamChangeCallback const& callback) {
        this->change_callback = callback;
    }

protected:
    /**
     * Ctor.
     */
    AbstractParam(void);

    /**
     * Answers whether this parameter object is assigned to a public slot.
     *
     * @return 'true' if this parameter object is assigned to a public
     *         slot, 'false' otherwise.
     */
    bool isSlotPublic(void) const;

    /**
     * Set has_changed flag to true.
     */
    void indicateChange() {
        has_changed = true;
        change_callback(slot);
    }

private:
    /** The holding slot */
    class AbstractParamSlot* slot;

    /**
     * Hash indicating fundamental changes in parameter definition
     * (i.e. requires rebuilding the UI).
     */
    uint64_t hash;

    /**
     * Indicating that the value has changed.
     */
    bool has_changed;

    /**
     * The change callback is set by the MegaMol Graph/Frontend as a notification mechanism
     * to be made aware of module-driven or other parameters changes not made via the lua parameter setter function
     */
    ParamChangeCallback change_callback = [](auto*) {
        // needs default init for randomly created modules/params not to crash for default SetValue() calls
    };
};


} /* end namespace param */
} /* end namespace core */
} /* end namespace megamol */

#endif /* MEGAMOLCORE_ABSTRACTPARAM_H_INCLUDED */
