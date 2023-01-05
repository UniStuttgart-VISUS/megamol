/**
 * MegaMol
 * Copyright (c) 2008, MegaMol Dev Team
 * All rights reserved.
 */

#pragma once

#include <memory>

#include "ObjectDescription.h"

namespace megamol::core {
class Call;
}

namespace megamol::core::factories {

/**
 * Description class for calls
 */
class CallDescription : public ObjectDescription {
public:
    typedef ::std::shared_ptr<const CallDescription> ptr;

    /** Ctor. */
    CallDescription() = default;

    /** Dtor. */
    ~CallDescription() override = default;

    /**
     * Answer the class name of the module described.
     *
     * @return The class name of the module described.
     */
    const char* ClassName() const override = 0;

    /**
     * Gets a human readable description of the module.
     *
     * @return A human readable description of the module.
     */
    const char* Description() const override = 0;

    /**
     * Creates a new call object.
     *
     * @return The newly created call object.
     */
    virtual Call* CreateCall() const = 0;

    /**
     * Answer the number of functions used for this call.
     *
     * @return The number of functions used for this call.
     */
    virtual unsigned int FunctionCount() const = 0;

    /**
     * Answer the name of the function used for this call.
     *
     * @param idx The index of the function to return it's name.
     *
     * @return The name of the requested function.
     */
    virtual const char* FunctionName(unsigned int idx) const = 0;

    /**
     * Answers whether this description is describing the class of
     * 'call'.
     *
     * @param call The call to test.
     *
     * @return 'true' if 'call' is described by this description,
     *         'false' otherwise.
     */
    virtual bool IsDescribing(const Call* call) const = 0;

    /**
     * Assignment crowbar
     *
     * @param tar The targeted object
     * @param src The source object
     */
    virtual void AssignmentCrowbar(Call* tar, const Call* src) const = 0;

protected:
    /**
     * Describes the call 'call'. Must be called for all calles created by
     * their describtion object before they are returned by 'CreateCall'.
     *
     * @param call The call to be described.
     *
     * @return 'call'
     */
    Call* describeCall(Call* call) const;
};

} // namespace megamol::core::factories
