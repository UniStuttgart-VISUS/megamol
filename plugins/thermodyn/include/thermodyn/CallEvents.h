#pragma once

#include <vector>

#include "mmcore/CallGeneric.h"
#include "mmcore/factories/CallAutoDescription.h"

#include "glm/glm.hpp"

namespace megamol::thermodyn {

class CallEvents : public core::GenericVersionedCall<std::shared_ptr<std::vector<glm::vec4>>, core::Spatial3DMetaData> {
public:
    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char* ClassName(void) {
        return "CallEvents";
    }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char* Description(void) {
        return "CallEvents.";
    }

    static unsigned int FunctionCount() {
        return core::GenericVersionedCall<std::shared_ptr<std::vector<glm::vec4>>,
            core::Spatial3DMetaData>::FunctionCount();
    }

    static const char* FunctionName(unsigned int idx) {
        return core::GenericVersionedCall<std::shared_ptr<std::vector<glm::vec4>>,
            core::Spatial3DMetaData>::FunctionName(idx);
    }
};

using CallEventsDescription = core::factories::CallAutoDescription<CallEvents>;

} // namespace megamol::thermodyn
