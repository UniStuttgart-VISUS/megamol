/**
 * MegaMol
 * Copyright (c) 2022, MegaMol Dev Team
 * All rights reserved.
 */

#pragma once

#include <chrono>
#include <fstream>
#include <iostream>
#include <mutex>
#include <string>

namespace megamol::ImageSeries::util {

class PerfTimer {

public:
    PerfTimer(std::string name, std::string filename) : line(name + ":" + filename) {
        start = std::chrono::high_resolution_clock::now();
    }

    ~PerfTimer() {
        try {
            writeToLog(line, std::chrono::high_resolution_clock::now() - start);
        } catch (...) {}
    }

private:
    std::string line;
    std::chrono::high_resolution_clock::time_point start;

    static inline void writeToLog(const std::string& line, std::chrono::high_resolution_clock::duration dur) {
        static std::ofstream stream = std::ofstream("perf.log", std::ios::trunc);
        static std::mutex mutex;
        std::scoped_lock<std::mutex> lock(mutex);
        stream << line << "=" << std::chrono::duration_cast<std::chrono::nanoseconds>(dur).count() << "\n";
    }
};

} // namespace megamol::ImageSeries::util
