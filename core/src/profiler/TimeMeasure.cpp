/*
 * TimeMeasure.cpp
 *
 * Copyright (C) 2021 by Universitaet Stuttgart (VISUS).
 * All rights reserved.
 */

#include "TimeMeasure.h"

using namespace megamol::core;

void profiler::Timer::startTimer() {
    this->start = std::chrono::steady_clock::now();
}

void profiler::Timer::endTimer() {
    this->end = std::chrono::steady_clock::now();
}

double profiler::Timer::getDuration(TimeUnit unit) {
    auto dur = (this->end - this->start);

    switch (unit) {
        case TimeUnit::SECONDS:
            this->duration = std::chrono::duration_cast<std::chrono::nanoseconds>(dur).count();
            break;
        case TimeUnit::MILLISECONDS:
            this->duration = std::chrono::duration_cast<std::chrono::milliseconds>(dur).count();
            break;
        case TimeUnit::MICROSECONDS:
            this->duration = std::chrono::duration_cast<std::chrono::microseconds>(dur).count();
            break;
        case TimeUnit::NANOSECONDS:
            this->duration = std::chrono::duration_cast<std::chrono::nanoseconds>(dur).count();
            break;
        default:
            this->duration = std::chrono::duration_cast<std::chrono::nanoseconds>(dur).count();
            break;
    }

    return 1.0;
}
