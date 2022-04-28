/*
 * AbstractParamSlot.h
 *
 * Copyright (C) 2008 by Universitaet Stuttgart (VISUS).
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_ABSTRACTPARAMSLOT_H_INCLUDED
#define MEGAMOLCORE_ABSTRACTPARAMSLOT_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "AbstractParam.h"
#include "vislib/SmartPtr.h"


namespace megamol {
namespace core {
namespace param {


/**
 * Abstract base class for all parameter slot classes
 */
class AbstractParamSlot {
public:
    friend class AbstractParam;

    /**
     * Dtor.
     */
    virtual ~AbstractParamSlot(void);

    /**
     * Answers whether this parameter is dirty (updated) or not.
     *
     * @return 'true' if the parameter is dirty (updated),
     *         'false' otherwise.
     */
    inline bool IsDirty(void) const {
        return this->dirty;
    }

    /**
     * Gets a pointer to the parameter of the slot casted to the specified
     * class 'C'. Do not free the returned pointer!
     *
     * @return A pointer to the parameter object of the slot.
     */
    template<class C>
    C* Param(void) {
        return this->param.DynamicCast<C>();
    }

    /**
     * Gets a pointer to the parameter of the slot casted to the specified
     * class 'C'. Do not free the returned pointer!
     *
     * @return A pointer to the parameter object of the slot.
     */
    template<class C>
    const C* Param(void) const {
        return this->param.DynamicCast<C>();
    }

    /**
     * Gets the parameter of the slot.
     *
     * @return The parameter of the slot.
     */
    inline vislib::SmartPtr<AbstractParam> Parameter(void) {
        return this->param;
    }

    /**
     * Gets the parameter of the slot.
     *
     * @return The parameter of the slot.
     */
    inline const vislib::SmartPtr<AbstractParam>& Parameter(void) const {
        return this->param;
    }

    /**
     * Resets the dirty flag.
     */
    inline void ResetDirty(void) {
        this->dirty = false;
    }

    /**
     * Sets the parameter object of this slot. This can only be done
     * before the slot has been made public! This must be done before the
     * slot can be made public, otherwise the program will assert.
     *
     * @param param The parameter object for this slot. The slot object
     *              takes ownership of the specified parameter object. You
     *              must not manipulate (especially 'free') the object
     *              after you called this method.
     */
    void SetParameter(AbstractParam* param);

    /**
     * Sets the parameter object of this slot. This can only be done
     * before the slot has been made public! This must be done before the
     * slot can be made public, otherwise the program will assert.
     *
     * @param param The parameter object for this slot. The slot object
     *              takes ownership of the specified parameter object. You
     *              must not manipulate (especially 'free') the object
     *              after you called this method.
     */
    inline void operator<<(AbstractParam* param) {
        this->SetParameter(param);
    }

    /**
     * Sets the dirty flag.
     */
    inline void ForceSetDirty(void) {
        this->update();
    }

protected:
    /**
     * Ctor.
     */
    AbstractParamSlot(void);

    /**
     * Answers whether the parameter member has already been set.
     *
     * @return 'true' if the parameter member has been set.
     */
    inline bool isParamSet(void) const {
        return this->param != NULL;
    }

    /**
     * Answers whether this slot has already been made available. If this
     * is the case, the initialisation phase has already been closed, and
     * no futher initialisation operations must take place.
     *
     * @return 'true' if the slot has already been made available.
     */
    virtual bool isSlotAvailable(void) const = 0;

    /**
     * Sets the dirty flag.
     */
    virtual void update(void);

private:
    /** The slots dirty flag */
    bool dirty;

#ifdef _WIN32
#pragma warning(disable : 4251)
#endif /* _WIN32 */
    /** The slots parameter */
    vislib::SmartPtr<AbstractParam> param;
#ifdef _WIN32
#pragma warning(default : 4251)
#endif /* _WIN32 */
};


} /* end namespace param */
} /* end namespace core */
} /* end namespace megamol */

#endif /* MEGAMOLCORE_ABSTRACTPARAMSLOT_H_INCLUDED */
