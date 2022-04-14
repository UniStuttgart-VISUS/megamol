/**
 * MegaMol
 * Copyright (c) 2021, MegaMol Dev Team
 * All rights reserved.
 */

#pragma once

#include <chrono>

namespace megamol::core::profiler {

class Timer {
public:
    enum class TimeUnit { NANOSECONDS = 0, MICROSECONDS = 1, MILLISECONDS = 2, SECONDS = 3 };

    void startTimer();
    void endTimer();
    double getDuration(TimeUnit unit);

private:
    std::chrono::steady_clock::time_point m_start;
    std::chrono::steady_clock::time_point m_end;
};

} // namespace megamol::core::profiler
