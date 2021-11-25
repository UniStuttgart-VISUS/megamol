#pragma once

#include <algorithm>
#include <array>
#include <cstdint>

namespace megamol {
namespace core {

/**
 * Class encapsulating the results captured from single performance region. It contains a ring buffer of length
 * buffer_length for keeping the values.
 */
class PerformanceHistory {
public:
    static constexpr uint32_t buffer_length = 100;

    PerformanceHistory();

    void push_value(double val);

    void reset();

    double operator[](int index) const;

    double last_value() const {
        return time_buffer[offset(next_index, buffer_length - 1)];
    }

    double average() const {
        return avg_time;
    }

    double buffer_average() const {
        return window_avg;
    }

    uint32_t samples() const {
        return num_samples;
    }

    // copies the disjunct segments in the ring buffer into a contiguous array for draw calls
    std::array<double, buffer_length> copyHistory() const {
        std::array<double, buffer_length> ret{};
        std::copy(time_buffer.begin() + next_index, time_buffer.end(), ret.begin());
        if (next_index > 0) {
            std::copy_n(time_buffer.begin(), next_index, ret.begin() + (buffer_length - next_index));
        }
        return ret;
    }

private:
    static int offset(const int index, const int offset);

    static int next_wrap(const int index) {
        return offset(index, 1);
    }
    static int prev_wrap(const int index) {
        return offset(index, -1);
    }

    std::array<double, buffer_length> time_buffer{};
    int next_index = 0;
    double avg_time = 0;
    double window_total = 0;
    double window_avg = 0;
    uint32_t num_samples = 0;
};

} // namespace core
} // namespace megamol
