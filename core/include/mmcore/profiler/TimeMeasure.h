/**
 * MegaMol
 * Copyright (c) 2022, MegaMol Dev Team
 * All rights reserved.
 */

#pragma once

#include <chrono>

namespace megamol {
namespace core {
namespace profiler {

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

} // namespace profiler
} // namespace core
} // namespace megamol
