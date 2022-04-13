/**
 * MegaMol
 * Copyright (c) 2021, MegaMol Dev Team
 * All rights reserved.
 */

#include "mmcore/profiler/TimeMeasure.h"

using namespace megamol::core;

void profiler::Timer::startTimer() {
    m_start = std::chrono::high_resolution_clock::now();
}

void profiler::Timer::endTimer() {
    m_end = std::chrono::high_resolution_clock::now();
}

double profiler::Timer::getDuration(TimeUnit unit) {
    double duration = 0.0;

    switch (unit) {
    case TimeUnit::SECONDS:
        duration = std::chrono::duration<double, std::milli>(m_end - m_start).count() * 1000.0;
        break;
    case TimeUnit::MILLISECONDS:
        duration = std::chrono::duration<double, std::milli>(m_end - m_start).count();
        break;
    case TimeUnit::MICROSECONDS:
        duration = std::chrono::duration<double, std::micro>(m_end - m_start).count();
        break;
    case TimeUnit::NANOSECONDS:
        duration = std::chrono::duration<double, std::nano>(m_end - m_start).count();
        break;
    default:
        duration = std::chrono::duration<double, std::nano>(m_end - m_start).count();
        break;
    }

    return duration;
}
