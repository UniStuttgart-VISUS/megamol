/**
 * MegaMol
 * Copyright (c) 2008, MegaMol Dev Team
 * All rights reserved.
 */

#pragma once

#include "mmcore/AbstractCallSlotPresentation.h"
#include "mmcore/AbstractSlot.h"
#include "mmcore/Call.h"
#include "mmcore/factories/CallDescription.h"
#include "mmcore/factories/CallDescriptionManager.h"
#include "vislib/Array.h"
#include "vislib/IllegalParamException.h"
#include "vislib/IllegalStateException.h"

namespace megamol::core {

/** Forward declaration */
class Module;


/**
 * A slot connection a Call to member pointers of a module.
 */
class CalleeSlot : public AbstractSlot, public AbstractCallSlotPresentation {
public:
    /**
     * Ctor.
     *
     * @param name The name of this slot.
     * @param desc A human readable description of this slot.
     */
    CalleeSlot(const vislib::StringA& name, const vislib::StringA& desc);

    /** Dtor. */
    virtual ~CalleeSlot();

    /**
     * Connects a call to this slot.
     *
     * @param call The call to connect.
     *
     * @return 'true' on success, 'false' on failure
     */
    bool ConnectCall(megamol::core::Call* call, factories::CallDescription::ptr call_description = nullptr);

    /**
     * Do not call this method directly!
     *
     * @param func The id of the function to be called.
     * @param call The call object calling this function.
     *
     * @return The return value of the function.
     */
    bool InCall(unsigned int func, Call& call);

    /**
     * Clears the cleanup mark for this and all dependent objects.
     */
    virtual void ClearCleanupMark();

    /**
     * Answers whether a given call is compatible with this slot.
     *
     * @param desc The description of the call to test.
     *
     * @return 'true' if the call is compatible, 'false' otherwise.
     */
    bool IsCallCompatible(factories::CallDescription::ptr desc) const override {
        if (desc == NULL)
            return false;
        vislib::StringA cn(desc->ClassName());

        for (unsigned int i = 0; i < desc->FunctionCount(); i++) {
            bool found = false;
            vislib::StringA fn(desc->FunctionName(i));
            for (unsigned int j = 0; j < this->callbacks.Count(); j++) {
                if (cn.Equals(this->callbacks[j]->CallName(), false) &&
                    fn.Equals(this->callbacks[j]->FuncName(), false)) {
                    found = true;
                    break;
                }
            }

            if (!found)
                return false;
        }

        return true;
    }

    /**
     * Registers the member 'func' as callback function for call 'callName'
     * function 'funcName'.
     *
     * @param callName The class name of the call.
     * @param funcName The name of the function of the call to register
     *                 this callback for.
     * @param func The member function pointer of the method to be used as
     *             callback. Use the class of the method as template
     *             parameter 'C'.
     */
    template<class C>
    void SetCallback(const char* callName, const char* funcName, bool (C::*func)(Call&)) {
        if (this->GetStatus() != AbstractSlot::STATUS_UNAVAILABLE) {
            throw vislib::IllegalStateException("You may not register "
                                                "callbacks after the slot has been enabled.",
                __FILE__, __LINE__);
        }

        vislib::StringA cn(callName);
        vislib::StringA fn(funcName);
        for (unsigned int i = 0; i < this->callbacks.Count(); i++) {
            if (cn.Equals(this->callbacks[i]->CallName(), false) && fn.Equals(this->callbacks[i]->FuncName(), false)) {
                throw vislib::IllegalParamException("callName funcName", __FILE__, __LINE__);
            }
        }
        Callback* cb = new CallbackImpl<C>(callName, funcName, func);
        this->callbacks.Add(cb);
    }

    /**
     * Registers the member 'func' as callback function for call 'callName'
     * function 'funcName'.
     *
     * @param callName The class name of the call.
     * @param funcName The name of the function of the call to register
     *                 this callback for.
     * @param obj The object of 'func'
     * @param func The member function pointer of the method to be used as
     *             callback. Use the class of the method as template
     *             parameter 'C'.
     */
    template<class C>
    void SetCallback(const char* callName, const char* funcName, C* obj, bool (C::*func)(Call&)) {
        if (this->GetStatus() != AbstractSlot::STATUS_UNAVAILABLE) {
            throw vislib::IllegalStateException("You may not register "
                                                "callbacks after the slot has been enabled.",
                __FILE__, __LINE__);
        }

        vislib::StringA cn(callName);
        vislib::StringA fn(funcName);
        for (unsigned int i = 0; i < this->callbacks.Count(); i++) {
            if (cn.Equals(this->callbacks[i]->CallName(), false) && fn.Equals(this->callbacks[i]->FuncName(), false)) {
                throw vislib::IllegalParamException("callName funcName", __FILE__, __LINE__);
            }
        }
        Callback* cb = new CallbackParentImpl<C>(callName, funcName, obj, func);
        this->callbacks.Add(cb);
    }

    /**
     * Answer the number of registered callback with names
     *
     * @return The number of registered callback with names
     */
    inline SIZE_T GetCallbackCount() const {
        return this->callbacks.Count();
    }

    /**
     * Answer the call class name of the idx-th registered callback
     *
     * @param idx The zero-based index of the callback to return the call class name of
     *
     * @return The call class name of the requested callback
     */
    inline const char* GetCallbackCallName(SIZE_T idx) const {
        return this->callbacks[idx]->CallName();
    }

    /**
     * Answer the function name of the idx-th registered callback
     *
     * @param idx The zero-based index of the callback to return the function name of
     *
     * @return The function name of the requested callback
     */
    inline const char* GetCallbackFuncName(SIZE_T idx) const {
        return this->callbacks[idx]->FuncName();
    }

private:
    /**
     * Nested base class for callback storage
     */
    class Callback {
    public:
        /**
         * Ctor
         *
         * @param callName The class name of the call.
         * @param funcName The name of the function.
         */
        Callback(const char* callName, const char* funcName) : callName(callName), funcName(funcName) {
            // intentionally empty
        }

        /** Dtor */
        virtual ~Callback() {
            // intentionally empty
        }

        /**
         * Call this callback.
         *
         * @param owner The owning object.
         * @param call The calling call.
         *
         * @return The return value of the function.
         */
        virtual bool CallMe(Module* owner, Call& call) = 0;

        /**
         * Gets the call class name.
         *
         * @return The call class name.
         */
        inline const char* CallName() const {
            return this->callName;
        }

        /**
         * Gets the function name.
         *
         * @return The function name.
         */
        inline const char* FuncName() const {
            return this->funcName;
        }

    private:
        /** the class name of the call */
        vislib::StringA callName;

        /** the name of the function */
        vislib::StringA funcName;
    };

    /**
     * Nested class for callback storage
     */
    template<class C>
    class CallbackImpl : public Callback {
    public:
        /**
         * Ctor
         *
         * @param func The callback member of 'C'
         */
        CallbackImpl(const char* callName, const char* funcName, bool (C::*func)(Call&))
                : Callback(callName, funcName)
                , func(func) {
            // intentionally empty
        }

        /** Dtor. */
        virtual ~CallbackImpl() {
            // intentionally empty
        }

        /**
         * Call this callback.
         *
         * @param owner The owning object.
         * @param call The calling call.
         *
         * @return The return value of the function.
         */
        virtual bool CallMe(Module* owner, Call& call) {
            C* c = dynamic_cast<C*>(owner);
            if (c == NULL)
                return false;
            return (c->*func)(call);
        }

    private:
        /** The callback method */
        bool (C::*func)(Call&);
    };

    /**
     * Nested class for callback storage
     */
    template<class C>
    class CallbackParentImpl : public Callback {
    public:
        /**
         * Ctor
         *
         * @param func The callback member of 'C'
         */
        CallbackParentImpl(const char* callName, const char* funcName, C* parent, bool (C::*func)(Call&))
                : Callback(callName, funcName)
                , func(func)
                , parent(parent) {
            // intentionally empty
        }

        /** Dtor. */
        virtual ~CallbackParentImpl() {
            // intentionally empty
        }

        /**
         * Call this callback.
         *
         * @param owner The owning object.
         * @param call The calling call.
         *
         * @return The return value of the function.
         */
        virtual bool CallMe(Module* owner, Call& call) {
            return (this->parent->*this->func)(call);
        }

    private:
        /** The callback method */
        bool (C::*func)(Call&);

        C* parent;
    };

    /** The registered callbacks */
    vislib::Array<Callback*> callbacks;
};

} // namespace megamol::core
