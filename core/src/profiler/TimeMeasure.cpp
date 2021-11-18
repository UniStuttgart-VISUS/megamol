/*
 * TimeMeasure.cpp
 *
 * Copyright (C) 2021 by Universitaet Stuttgart (VISUS).
 * All rights reserved.
 */

#include "mmcore/profiler/TimeMeasure.h"

using namespace megamol::core;

void profiler::Timer::startTimer() {
    m_start = std::chrono::steady_clock::now();
}

void profiler::Timer::endTimer() {
    m_end = std::chrono::steady_clock::now();
}

double profiler::Timer::getDuration(TimeUnit unit) {
    auto dur = (m_end - m_start);

    switch (unit) {
        case TimeUnit::SECONDS:
        m_duration = (double) std::chrono::duration_cast<std::chrono::nanoseconds>(m_end - m_start).count() / 1e9f;
            break;
        case TimeUnit::MILLISECONDS:
            m_duration = (double) std::chrono::duration_cast<std::chrono::nanoseconds>(m_end - m_start).count() / 1e6f;
            break;
        case TimeUnit::MICROSECONDS:
            m_duration = (double) std::chrono::duration_cast<std::chrono::nanoseconds>(m_end - m_start).count() / 1e3f;
            break;
        case TimeUnit::NANOSECONDS:
            m_duration = (double) std::chrono::duration_cast<std::chrono::nanoseconds>(m_end - m_start).count();
            break;
        default:
            m_duration = (double) std::chrono::duration_cast<std::chrono::nanoseconds>(m_end - m_start).count();
            break;
    }


    return m_duration;
}
