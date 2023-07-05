#pragma once

#include <functional>

namespace megamol::frontend_resources {

static std::string PowerCallbacks_Req_Name = "PowerCallbacks";

struct PowerCallbacks {
    std::function<unsigned long()> signal_high;
    std::function<unsigned long()> signal_low;
};

} // namespace megamol::frontend_resources
