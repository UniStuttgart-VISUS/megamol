/*
 * TimeMeasure.h
 *
 * Copyright (C) 2021 by Universitaet Stuttgart (VISUS).
 * All rights reserved.
 */

#ifndef TIME_MEASURE_H_INCLUDED
#define TIME_MEASURE_H_INCLUDED

#include <chrono>

namespace megamol {
namespace core {
namespace profiler {

    class Timer {
    public:
        enum class TimeUnit
        {
            NANOSECONDS = 0,
            MICROSECONDS = 1,
            MILLISECONDS = 2,
            SECONDS = 3
        };

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

#endif // TIME_MEASURE_H_INCLUDED
