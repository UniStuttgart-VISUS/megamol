#include "mmcore/MultiPerformanceHistory.h"

#include <cassert>
#include <iostream>

using namespace megamol::core;

MultiPerformanceHistory::MultiPerformanceHistory() {
    reset();
}

void MultiPerformanceHistory::push_sample(frame_type frame, frame_index_type idx, perf_type val) {

    // general question: window_total might be prone to drift, but the average of many values
    // every frame for many callbacks is much too expensive, isn't it?

    if (idx == 0) {
        // before we advance, use the frame for full-window statistics
        auto& buf = time_buffer[next_index];
        window_metrics[static_cast<uint32_t>(metric_type::MIN)].push_value(buf.min());
        window_metrics[static_cast<uint32_t>(metric_type::MAX)].push_value(buf.max());
        window_metrics[static_cast<uint32_t>(metric_type::AVERAGE)].push_value(buf.avg());
        window_metrics[static_cast<uint32_t>(metric_type::MEDIAN)].push_value(buf.med());
        window_metrics[static_cast<uint32_t>(metric_type::COUNT)].push_value(buf.count());
        window_metrics[static_cast<uint32_t>(metric_type::SUM)].push_value(buf.sum());
        next_index = next_wrap(next_index);
        num_frames++;
    }
    time_buffer[next_index].push_value(frame, val);
    num_samples++;
}

void MultiPerformanceHistory::reset() {
    next_index = 0;
    for (auto& tb : time_buffer) {
        tb.reset();
    }
    num_samples = 0;
    num_frames = 0;
}

uint32_t MultiPerformanceHistory::samples(int index) {
    return time_buffer[offset(next_index, index)].count();
}

MultiPerformanceHistory::perf_type MultiPerformanceHistory::at(int index, metric_type metric) const {
    switch (metric) {
    case metric_type::MIN:
        return time_buffer[offset(next_index, index)].min();
    case metric_type::MAX:
        return time_buffer[offset(next_index, index)].max();
    case metric_type::AVERAGE:
        return time_buffer[offset(next_index, index)].avg();
    case metric_type::MEDIAN:
        return time_buffer[offset(next_index, index)].med();
    case metric_type::COUNT:
        return time_buffer[offset(next_index, index)].count();
    case metric_type::SUM:
        return time_buffer[offset(next_index, index)].sum();
    }
    return perf_type();
}

MultiPerformanceHistory::perf_type MultiPerformanceHistory::last_value(metric_type metric) const {
    return at(buffer_length - 1, metric);
}

std::array<MultiPerformanceHistory::perf_type, MultiPerformanceHistory::buffer_length>
MultiPerformanceHistory::copyHistory(metric_type metric) const {
    std::array<perf_type, buffer_length> ret{};
    auto trafo = [metric](const frame_statistics& fs) {
        switch (metric) {
        case metric_type::MIN:
            return fs.min();
        case metric_type::MAX:
            return fs.max();
        case metric_type::AVERAGE:
            return fs.avg();
        case metric_type::MEDIAN:
            return fs.med();
        case metric_type::COUNT:
            return static_cast<MultiPerformanceHistory::perf_type>(fs.count());
        case metric_type::SUM:
            return fs.sum();
        }
        return float();
    };
    std::transform(time_buffer.begin() + next_index, time_buffer.end(), ret.begin(), trafo);
    if (next_index > 0) {
        //std::copy_n(time_buffer.begin(), next_index, ret.begin() + (buffer_length - next_index));
        std::transform(
            time_buffer.begin(), time_buffer.begin() + next_index, ret.begin() + (buffer_length - next_index), trafo);
    }
    return ret;
}

int MultiPerformanceHistory::offset(const int index, const int offset) {
    auto o = offset % buffer_length;
    return (index + offset + buffer_length) % buffer_length;
}
