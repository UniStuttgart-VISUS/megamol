#include "mmcore/PerformanceHistory.h"

using namespace megamol::core;

PerformanceHistory::PerformanceHistory() {
    reset();
}

void PerformanceHistory::push_value(double val) {

    // general question: window_total might be prone to drift, but the average of many values
    // every frame for many callbacks is much too expensive, isn't it?

    // remove the window sum component that is going to be overwritten
    // array starts zeroed so unused samples do not change the result here
    window_total -= time_buffer[next_index];
    window_total += val;

    time_buffer[next_index] = val;
    const auto total = avg_time * num_samples + val;
    num_samples++;
    avg_time = total / static_cast<double>(num_samples);

    // until we have at least a sample everywhere, the average is over num_samples only
    window_avg = window_total / std::min(buffer_length, num_samples);

    next_index = next_wrap(next_index);
}

void PerformanceHistory::reset() {
    next_index = 0;
    time_buffer.fill(0.0);
    num_samples = 0;
    avg_time = 0;
    window_total = 0;
    window_avg = 0;
}

double PerformanceHistory::operator[](int index) const {
    return time_buffer[offset(next_index, index)];
}

int PerformanceHistory::offset(const int index, const int offset) {
    auto o = offset % buffer_length;
    return (index + offset + buffer_length) % buffer_length;
}
