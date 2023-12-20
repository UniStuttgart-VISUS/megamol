#pragma once

#include <functional>
#include <string>

namespace megamol::frontend_resources {
static std::string FrameStatsCallbacks_Req_Name = "FrameStatsCallbacks";

struct FrameStatsCallbacks {
    std::function<void()> mark_frame;
};
} // namespace megamol::frontend_resources
