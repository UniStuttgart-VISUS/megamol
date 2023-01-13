/**
 * MegaMol
 * Copyright (c) 2008, MegaMol Dev Team
 * All rights reserved.
 */

#pragma once

#include <memory>
#include <vector>

#include "mmcore/AbstractCallSlotPresentation.h"
#include "mmcore/AbstractSlot.h"
#include "mmcore/Call.h"
#include "mmcore/factories/CallDescription.h"
#include "vislib/String.h"
#include "vislib/macro_utils.h"

namespace megamol::core {

/** forward declaration */
class CalleeSlot;


/**
 * The caller slot connect one calling module with one called module
 * using a Call object.
 */
class CallerSlot : public AbstractSlot, public AbstractCallSlotPresentation {
public:
    /**
     * Ctor.
     *
     * @param name The name of this slot (needs to be unqiue per module).
     * @param desc A human readable description of this slot.
     */
    CallerSlot(const vislib::StringA& name, const vislib::StringA& desc);

    /** Dtor. */
    virtual ~CallerSlot();

    /**
     * Calls the call function 'func'.
     *
     * @param func The function to be called.
     *
     * @return 'true' on success, 'false' on failure.
     */
    bool Call(unsigned int func = 0);

    /**
     * Gets the call dynamic-casted to the specified template type.
     *
     * @return The casted call or 'NULL' in case the slot is not connected
     *         or the call could not be casted.
     */
    template<class T>
    inline T* CallAs() {
        return dynamic_cast<T*>(this->call);
    }

    /**
     * Connects a call to this slot. The caller slot takes the memory
     * ownership of the call object.
     *
     * @param call The call to connect.
     *
     * @return 'true' on success, 'false' on failure
     */
    bool ConnectCall(megamol::core::Call* call) {
        vislib::sys::AbstractReaderWriterLock* lock = NULL;
        if (this->Parent()) {
            lock =
                &this->Parent()
                     ->ModuleGraphLock(); // TODO << This is a potential HAZARD!!!! Can lock ModuleGraph while ModuleGraph is already locked
            lock->LockExclusive();
        }

        if (call == NULL) {
            if (this->call != NULL) {
                this->SetStatusDisconnected();
                //megamol::core::Call *c = this->call;
                this->call = NULL;
                //delete c;
            }
            if (lock != NULL)
                lock->UnlockExclusive();
            return true;
        }

        for (unsigned int i = 0; i < this->compDesc.size(); i++) {
            if (this->compDesc[i]->IsDescribing(call)) {
                if (this->call != NULL)
                    this->call->caller = NULL;
                //delete this->call;
                this->call = call;
                this->call->caller = this;
                this->SetStatusConnected();
                if (lock != NULL)
                    lock->UnlockExclusive();
                return true;
            }
        }

        if (lock != NULL)
            lock->UnlockExclusive();
        return false;
    }

    bool IsCallCompatible(std::string const& call_class_name) const {
        vislib::StringA dcn = call_class_name.c_str();

        for (unsigned int i = 0; i < this->compDesc.size(); i++) {
            if (dcn.Equals(this->compDesc[i]->ClassName())) {
                return true;
            }
        }
        return false;
    }

    /**
     * Answers whether a given call is compatible with this slot.
     *
     * @param desc The description of the call to test.
     *
     * @return 'true' if the call is compatible, 'false' otherwise.
     */
    bool IsCallCompatible(factories::CallDescription::ptr desc) const override {
        if (!desc)
            return false;

        return IsCallCompatible(std::string{desc->ClassName()});
    }

    /**
     * Registers a call as compatible call with this caller slot. Use the
     * description class of the call as template parameter 'T'. Calls of
     * this class can then be connected to this slot.
     */
    template<class T>
    inline void SetCompatibleCall() {
        factories::CallDescription::ptr d = std::make_shared<T>();
        for (unsigned int i = 0; i < this->compDesc.size(); i++) {
            if (vislib::StringA(this->compDesc[i]->ClassName()).Equals(d->ClassName())) {
                return;
            }
        }
        this->compDesc.push_back(d);
    }

    /**
     * Registers a call as compatible call with this caller slot. Calls of
     * this class can then be connected to this slot.
     *
     * @param desc The description object to be added.
     */
    inline void SetCompatibleCall(factories::CallDescription::ptr desc) {
        assert(desc);
        for (unsigned int i = 0; i < this->compDesc.size(); i++) {
            if (vislib::StringA(this->compDesc[i]->ClassName()).Equals(desc->ClassName())) {
                return;
            }
        }
        this->compDesc.push_back(desc);
    }

    /**
     * Answers whether this caller is connected to the specified callee.
     *
     * @param target The callee slot
     *
     * @return The connecting call or 'NULL' if 'this' and 'target' are not connected.
     */
    ::megamol::core::Call* IsConnectedTo(::megamol::core::CalleeSlot* target);

    /**
     * Answers whether this caller is connected to the specified callee.
     *
     * @param target The callee slot
     *
     * @return The connecting call or 'NULL' if 'this' and 'target' are not connected.
     */
    const ::megamol::core::Call* IsConnectedTo(const ::megamol::core::CalleeSlot* target) const;

    /**
     * Clears the cleanup mark for this and all dependent objects.
     */
    virtual void ClearCleanupMark();

    /**
     * Disconnects calls from all slots which are marked for cleanup.
     */
    virtual void DisconnectCalls();

    /**
     * TODO: Document me
     */
    inline SIZE_T GetCompCallCount() const {
        return this->compDesc.size();
    }

    /**
     * TODO: Document me
     */
    inline const char* GetCompCallClassName(SIZE_T idx) const {
        return this->compDesc[idx]->ClassName();
    }

private:
    /** The connecting call object */
    megamol::core::Call* call;

    /** Array of descriptions of compatible calls. */
    VISLIB_MSVC_SUPPRESS_WARNING(4251)
    ::std::vector<factories::CallDescription::ptr> compDesc;
};


} // namespace megamol::core
