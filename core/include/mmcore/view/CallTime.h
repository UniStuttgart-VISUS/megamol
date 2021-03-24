#pragma once

#include "mmcore/CallGeneric.h"
#include "mmcore/factories/CallAutoDescription.h"

namespace megamol::core::view {
class CallTime : public GenericVersionedCall<uint64_t, EmptyMetaData> {
public:
    /**
     * Answer the name of the objects of this description.
     *
     * @return The name of the objects of this description.
     */
    static const char* ClassName(void) {
        return "CallTime";
    }

    /**
     * Gets a human readable description of the module.
     *
     * @return A human readable description of the module.
     */
    static const char* Description(void) {
        return "Call transporting time point";
    }

    /**
     * Answer the number of functions used for this call.
     *
     * @return The number of functions used for this call.
     */
    static unsigned int FunctionCount(void) {
        return 1;
    }

    /**
     * Answer the name of the function used for this call.
     *
     * @param idx The index of the function to return it's name.
     *
     * @return The name of the requested function.
     */
    static const char* FunctionName(unsigned int idx) {
        return GenericVersionedCall::FunctionName(idx);
    }

    CallTime() = default;

    virtual ~CallTime() = default;
};

using CallTimeDescription = factories::CallAutoDescription<CallTime>;

} // namespace megamol::core::view
