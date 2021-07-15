#pragma once

#include <string>
#include <vector>

#include "mmcore/CallGeneric.h"
#include "mmcore/factories/CallAutoDescription.h"

namespace megamol::thermodyn {
struct stats_info_element {
    std::string name;
    float mean;
    float stddev;
};

class CallStatsInfo : public core::GenericVersionedCall<std::vector<stats_info_element>, core::Spatial3DMetaData> {
public:
    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char* ClassName(void) {
        return "CallStatsInfo";
    }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char* Description(void) {
        return "Transports statistics on distribution functions.";
    }

    static unsigned int FunctionCount() {
        return core::GenericVersionedCall<std::vector<stats_info_element>, core::Spatial3DMetaData>::FunctionCount();
    }

    static const char* FunctionName(unsigned int idx) {
        return core::GenericVersionedCall<std::vector<stats_info_element>, core::Spatial3DMetaData>::FunctionName(idx);
    }
};

using CallStatsInfoDescription = core::factories::CallAutoDescription<CallStatsInfo>;

} // namespace megamol::thermodyn
