/**
 * MegaMol
 * Copyright (c) 2008, MegaMol Dev Team
 * All rights reserved.
 */

#pragma once

#include <memory>

#include "mmcore/AbstractSlot.h"
#include "mmcore/param/AbstractParamSlot.h"
#include "vislib/assert.h"

namespace megamol::core {

/** Forward declaration */
class Module;

namespace param {

/**
 * Class of parameter slots holding parameter object.
 *
 * Use the update callback or the dirty flag mechanism to receive the
 * information when the value of the parameter changes.
 */
class ParamSlot : public AbstractSlot, public AbstractParamSlot {
public:
    /**
     * Ctor.
     *
     * @param name The name of this slot.
     * @param desc A human readable description of this slot.
     */
    ParamSlot(const vislib::StringA& name, const vislib::StringA& desc);

    /** Dtor. */
    ~ParamSlot() override;

    /**
     * Makes this slot available. After this method was called the
     * settings of the slot can no longer be changed.
     */
    void MakeAvailable() override;

    /**
     * Sets an update callback method, which is called whenever the dirty
     * flag is set (the value of the parameter changes). The method will
     * not be called when the parameter changes, but the dirty flag still
     * is set!
     *
     * The callback method has one parameter: the slot the callback is
     * registered to and that is calling this callback.
     *
     * The return value controls the dirty flag (which is set before the
     * callback is called). If the method returns 'true' the dirty flag is
     * reset after the callback returns. If it returns 'false' the dirty
     * flag remains set.
     *
     * Be aware that the callback might be called from within another
     * thread.
     *
     * @param obj The callback object.
     * @param func The callback member function.
     */
    template<class C>
    void SetUpdateCallback(C* obj, bool (C::*func)(ParamSlot&)) {
        if (this->callback != nullptr) {
            delete this->callback;
        }
        this->callback = new CallbackImpl<C>(obj, func);
    }

    /**
     * ---
     * NOTE: Do not call this version if param slot is made available afterwards
     *     in child class. Call SetUpdateCallback(obj, func) in parent class instead.
     * ---
     *
     * Sets an update callback method, which is called whenever the dirty
     * flag is set (the value of the parameter changes). The method will
     * not be called when the parameter changes, but the dirty flag still
     * is set!
     *
     * The callback method has one parameter: the slot the callback is
     * registered to and that is calling this callback.
     *
     * The return value controls the dirty flag (which is set before the
     * callback is called). If the method returns 'true' the dirty flag is
     * reset after the callback returns. If it returns 'false' the dirty
     * flag remains set.
     *
     * Be aware that the callback might be called from within another
     * thread.
     *
     * @param func The callback member function.
     */
    template<class C>
    void SetUpdateCallback(bool (C::*func)(ParamSlot&)) {
        if (this->callback != nullptr) {
            delete this->callback;
        }
        this->callback = new CallbackImpl<C>(NULL, func);
    }

    /**
     * Queue a notification that the parameter value has changed, to notify
     * those that have a registered listener. This method is public to allow
     * pushing parameter changes that cannot use the dirty flag to avoid feedback
     * loops.
     *
     * @param force Enforce notification, otherwise the notification will only be triggered if the value has
     * changed.
     */
    void QueueUpdateNotification(bool force = false);

protected:
    /**
     * Answers whether this slot has already been made available. If this
     * is the case, the initialisation phase has already been closed, and
     * no futher initialisation operations must take place.
     *
     * @return 'true' if the slot has already been made available.
     */
    bool isSlotAvailable() const override;

private:
    /**
     * Nested base class for callback storage
     */
    class Callback {
    public:
        /** Ctor. */
        Callback() {
            // intentionally empty
        }

        /** Dtor. */
        virtual ~Callback() {
            // intentionally empty
        }

        /**
         * Triggers the update callback.
         *
         * @param owner The module owning the slot
         * @param slot The slot being updated and owning this callback.
         *
         * @return 'true' if the dirty flag should be resetted, 'false' if
         *         the dirty flag should remain set.
         */
        virtual bool Update(Module* owner, ParamSlot& slot) = 0;
    };

    /**
     * Nested class for callback storage
     */
    template<class C>
    class CallbackImpl : public Callback {
    public:
        /**
         * Ctor.
         *
         * @param obj The object of the callback function.
         * @param func The callback function meber.
         */
        CallbackImpl(C* obj, bool (C::*func)(ParamSlot&)) : Callback(), obj(obj), func(func) {
            ASSERT(func != NULL);
        }

        /** Dtor. */
        ~CallbackImpl() override {
            this->func = NULL; // DO NOT DELETE
        }

        /**
         * Triggers the update callback.
         *
         * @param owner The module owning the slot
         * @param slot The slot being updated and owning this callback.
         *
         * @return 'true' if the dirty flag should be resetted, 'false' if
         *         the dirty flag should remain set.
         */
        bool Update(Module* owner, ParamSlot& slot) override {
            return update(owner, slot);
        }

    private:
        template<typename ModuleT = Module>
        bool update(typename std::enable_if<std::is_base_of<ModuleT, C>::value, Module>::type* owner, ParamSlot& slot) {
            return ((this->obj == NULL) ? ((dynamic_cast<C*>(owner)->*this->func)(slot))
                                        : ((this->obj->*this->func)(slot)));
        }

        template<typename ModuleT = Module>
        bool update(typename std::enable_if<!std::is_base_of<ModuleT, C>::value, Module>::type*, ParamSlot& slot) {
            ASSERT(this->obj != NULL);
            return (this->obj->*this->func)(slot);
        }

        /** The callback object */
        C* obj;

        /** The callback member */
        bool (C::*func)(ParamSlot&);
    };

    /**
     * Sets the dirty flag and triggers the update callback if the dirty
     * flag was not set before.
     */
    void update() override;

    /** The update callback object */
    Callback* callback;
};


} /* end namespace param */
} // namespace megamol::core
