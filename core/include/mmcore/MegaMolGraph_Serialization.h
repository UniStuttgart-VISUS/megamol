#pragma once

#include <string>

namespace megamol::core {

    struct MegaMolGraph_Serialization {
        std::string serInstances;
        std::string serModules;
        std::string serCalls;
        std::string serParams;
    };

}; // namespace megamol::core
