/*
 * FrameStatistics.h
 *
 * Copyright (C) 2020 by VISUS (Universitaet Stuttgart).
 * Alle Rechte vorbehalten.
 */

#pragma once

#include <string>
#include <vector>

namespace megamol {
namespace frontend_resources {

static std::string FrameStatistics_Req_Name = "FrameStatistics";

struct FrameStatistics {
    double elapsed_program_time_seconds = 0.0;
    size_t rendered_frames_count = 0;

    double last_rendered_frame_time_milliseconds = 0.0;
    double last_averaged_fps = 0.0;
    double last_averaged_mspf = 0.0;
};

} /* end namespace frontend_resources */
} /* end namespace megamol */
