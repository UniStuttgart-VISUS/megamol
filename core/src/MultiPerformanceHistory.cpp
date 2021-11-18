#include "mmcore/MultiPerformanceHistory.h"

using namespace megamol::core;

MultiPerformanceHistory::MultiPerformanceHistory() {
    reset();
}

void MultiPerformanceHistory::push_sample(frame_type frame, perf_type val) {

    // general question: window_total might be prone to drift, but the average of many values
    // every frame for many callbacks is much too expensive, isn't it?

    auto& buf = time_buffer[next_index];
    if (frame != buf.frame()) {
        // before we advance, use the frame for full-window statistics
        // does that actually work?
        window_metrics[static_cast<uint32_t>(metric_type::MIN)].push_value(buf.frame(), buf.min());
        window_metrics[static_cast<uint32_t>(metric_type::MAX)].push_value(buf.frame(), buf.max());
        window_metrics[static_cast<uint32_t>(metric_type::AVERAGE)].push_value(buf.frame(), buf.avg());
        window_metrics[static_cast<uint32_t>(metric_type::MEDIAN)].push_value(buf.frame(), buf.med());
        window_metrics[static_cast<uint32_t>(metric_type::COUNT)].push_value(buf.frame(), buf.count());
        next_index = next_wrap(next_index);
        buf = time_buffer[next_index];
        num_frames++;
    }
    //// remove the window sum component that is going to be overwritten
    //// array starts zeroed so unused samples do not change the result here
    //window_averages_total -= buf.avg();
    buf.push_value(frame, val);
    //window_averages_total += buf.avg();
    num_samples++;

    // until we have at least a sample everywhere, the average is over num_samples only
    //window_average = window_averages_total / std::min(buffer_length, num_samples);

    //time_buffer[next_index] = val;
    //const auto total = avg_time * num_samples + val;
    //num_samples++;
    //avg_time = total / static_cast<double>(num_samples);

    //next_index = next_wrap(next_index);
}

void MultiPerformanceHistory::reset() {
    next_index = 0;
    for (auto& tb: time_buffer) {
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
        break;
    case metric_type::MAX:
        return time_buffer[offset(next_index, index)].max();
        break;
    case metric_type::AVERAGE:
        return time_buffer[offset(next_index, index)].avg();
        break;
    case metric_type::MEDIAN:
        return time_buffer[offset(next_index, index)].med();
        break;
    case metric_type::COUNT:
        return time_buffer[offset(next_index, index)].count();
        break;
    }
}

MultiPerformanceHistory::perf_type MultiPerformanceHistory::last_value(metric_type metric) const {
    return at(buffer_length - 1, metric);
}

std::array<MultiPerformanceHistory::perf_type, MultiPerformanceHistory::buffer_length> MultiPerformanceHistory::copyHistory(
    metric_type metric) const {
    std::array<perf_type, buffer_length> ret{};
    auto trafo = [metric](const frame_statistics& fs) {
        switch (metric) {
        case metric_type::MIN:
            return fs.min();
            break;
        case metric_type::MAX:
            return fs.max();
            break;
        case metric_type::AVERAGE:
            return fs.avg();
            break;
        case metric_type::MEDIAN:
            return fs.med();
            break;
        case metric_type::COUNT:
            return static_cast<MultiPerformanceHistory::perf_type>(fs.count());
            break;
        }
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
