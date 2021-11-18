#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace megamol {
namespace core {

/**
 * Class encapsulating the results captured from single performance region. It contains a ring buffer of length
 * buffer_length for keeping the values. Multiple values per frame a supported since a region can be entered
 * multiple times.
 */
class MultiPerformanceHistory {
public:
    static const uint32_t buffer_length = 100;
    using frame_type = uint32_t;
    using perf_type = float;

    enum class metric_type { MIN = 0, MAX = 1, AVERAGE = 2, MEDIAN = 3, COUNT = 4 };

    MultiPerformanceHistory();

    void set_name(std::string n) {
        name = n;
    }

    const std::string& get_name() const {
        return name;
    }

    void push_sample(frame_type frame, perf_type val);

    void reset();

    perf_type at(int index, metric_type metric) const;

    perf_type last_value(metric_type metric) const;

    perf_type average(metric_type metric) const;

    perf_type window_statistics(metric_type outer_metric, metric_type inner_metric) const {
        switch (outer_metric) {
        case metric_type::MIN:
            return window_metrics[static_cast<uint32_t>(inner_metric)].min();
        case metric_type::MAX:
            return window_metrics[static_cast<uint32_t>(inner_metric)].max();
        case metric_type::AVERAGE:
            return window_metrics[static_cast<uint32_t>(inner_metric)].avg();
        case metric_type::MEDIAN:
            return window_metrics[static_cast<uint32_t>(inner_metric)].med();
        case metric_type::COUNT:
            return window_metrics[static_cast<uint32_t>(inner_metric)].count();
        }
    }

    uint32_t samples() const {
        return num_samples;
    }
    uint32_t samples(int index);
    uint32_t frames() const {
        return num_frames;
    }

    // copies the disjunct segments in the ring buffer into a contiguous array for draw calls
    std::array<perf_type, buffer_length> copyHistory(metric_type metric) const;

private:
    static int offset(const int index, const int offset);

    static int next_wrap(const int index) {
        return offset(index, 1);
    }
    static int prev_wrap(const int index) {
        return offset(index, -1);
    }

    class frame_statistics {
    public:
        void push_value(frame_type f, perf_type t) {
            if (curr_frame != f) {
                reset();
                curr_frame = f;
            }
            values.push_back(t);
            if (t < minimum)
                minimum = t;
            if (t > maximum)
                maximum = t;
            const auto total = average * (values.size() - 1) + t;
            average = total / static_cast<perf_type>(values.size());
        }
        perf_type min() const {
            return minimum;
        };
        perf_type max() const {
            return maximum;
        }
        perf_type avg() const {
            return average;
        }
        uint32_t count() const {
            return values.size();
        }
        perf_type med() const {
            if (curr_frame != median_computed) {
                std::vector<perf_type> copy(values);
                std::sort(copy.begin(), copy.end());
                median = copy[copy.size() / 2];
                median_computed = curr_frame;
            }
            return median;
        }
        frame_type frame() const {
            return curr_frame;
        }
        void reset() {
            values.clear();
            average = median = 0;
            minimum = std::numeric_limits<perf_type>::max();
            maximum = std::numeric_limits<perf_type>::lowest();
        }

    private:
        std::vector<perf_type> values;
        perf_type minimum = std::numeric_limits<perf_type>::max(), maximum = std::numeric_limits<perf_type>::lowest();
        perf_type average = 0;
        mutable perf_type median = 0;
        frame_type curr_frame = std::numeric_limits<frame_type>::max();
        mutable frame_type median_computed = std::numeric_limits<frame_type>::max();
    };

    std::array<frame_statistics, buffer_length> time_buffer{};
    std::string name;
    int next_index = 0;
    uint32_t num_samples = 0, num_frames = 0;
    std::array<frame_statistics, 5> window_metrics;
};

}
}
