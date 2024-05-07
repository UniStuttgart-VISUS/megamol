/**
 * MegaMol
 * Copyright (c) 2020, MegaMol Dev Team
 * All rights reserved.
 */

#pragma once

#include <chrono>
#include <string>
#include <vector>

namespace megamol::frontend_resources {

static std::string FrameStatistics_Req_Name = "FrameStatistics";

struct FrameStatistics {
    std::chrono::milliseconds elapsed_program_time_milliseconds;
    std::size_t rendered_frames_count = 0;

    std::chrono::microseconds last_rendered_frame_time_microseconds;
    double last_averaged_fps = 0.0;
    double last_averaged_mspf = 0.0;
};

} // namespace megamol::frontend_resources
