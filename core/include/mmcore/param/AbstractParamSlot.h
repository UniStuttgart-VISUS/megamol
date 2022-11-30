/**
 * MegaMol
 * Copyright (c) 2008, MegaMol Dev Team
 * All rights reserved.
 */

#pragma once

#include <memory>

#include "AbstractParam.h"

namespace megamol::core::param {

/**
 * Abstract base class for all parameter slot classes
 */
class AbstractParamSlot {
public:
    friend class AbstractParam;

    /**
     * Dtor.
     */
    virtual ~AbstractParamSlot();

    /**
     * Answers whether this parameter is dirty (updated) or not.
     *
     * @return 'true' if the parameter is dirty (updated),
     *         'false' otherwise.
     */
    inline bool IsDirty() const {
        return this->dirty;
    }

    /**
     * Gets a pointer to the parameter of the slot casted to the specified
     * class 'C'. Do not free the returned pointer!
     *
     * @return A pointer to the parameter object of the slot.
     */
    template<class C>
    C* Param() {
        return dynamic_cast<C*>(this->param.get());
    }

    /**
     * Gets a pointer to the parameter of the slot casted to the specified
     * class 'C'. Do not free the returned pointer!
     *
     * @return A pointer to the parameter object of the slot.
     */
    template<class C>
    const C* Param() const {
        return dynamic_cast<C*>(this->param.get());
    }

    /**
     * Gets the parameter of the slot.
     *
     * @return The parameter of the slot.
     */
    inline std::shared_ptr<AbstractParam> Parameter() {
        return this->param;
    }

    /**
     * Gets the parameter of the slot.
     *
     * @return The parameter of the slot.
     */
    inline const std::shared_ptr<AbstractParam>& Parameter() const {
        return this->param;
    }

    /**
     * Resets the dirty flag.
     */
    inline void ResetDirty() {
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
    inline void ForceSetDirty() {
        this->update();
    }

protected:
    /**
     * Ctor.
     */
    AbstractParamSlot();

    /**
     * Answers whether the parameter member has already been set.
     *
     * @return 'true' if the parameter member has been set.
     */
    inline bool isParamSet() const {
        return this->param != nullptr;
    }

    /**
     * Answers whether this slot has already been made available. If this
     * is the case, the initialisation phase has already been closed, and
     * no futher initialisation operations must take place.
     *
     * @return 'true' if the slot has already been made available.
     */
    virtual bool isSlotAvailable() const = 0;

    /**
     * Sets the dirty flag.
     */
    virtual void update();

private:
    /** The slots dirty flag */
    bool dirty;

    /** The slots parameter */
    std::shared_ptr<AbstractParam> param;
};


} // namespace megamol::core::param
