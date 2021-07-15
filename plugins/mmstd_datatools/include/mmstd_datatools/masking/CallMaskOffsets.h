#pragma once

#include <memory>
#include <vector>

#include "mmcore/CallGeneric.h"
#include "mmcore/factories/CallAutoDescription.h"

namespace megamol::stdplugin::datatools::masking {
using offsets_t = std::vector<uint64_t>;

class CallMaskOffsets : public core::GenericVersionedCall<std::shared_ptr<offsets_t>, core::EmptyMetaData> {
public:
    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char* ClassName(void) {
        return "CallMaskOffsets";
    }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char* Description(void) {
        return "CallMaskOffsets";
    }
};

using CallMaskOffsetsDescription = core::factories::CallAutoDescription<CallMaskOffsets>;

} // namespace megamol::stdplugin::datatools::masking
