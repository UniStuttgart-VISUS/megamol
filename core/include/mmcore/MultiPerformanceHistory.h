/**
 * MegaMol
 * Copyright (c) 2021, MegaMol Dev Team
 * All rights reserved.
 */

#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <list>
#include <string>
#include <vector>

namespace megamol::core {

/**
 * Class encapsulating the results captured from a single performance region. It contains a ring buffer of length
 * buffer_length for keeping the values. Multiple values per frame a supported since a region can be entered
 * multiple times, for example when the graph activates a module more than once per frame.
 */
class MultiPerformanceHistory {
public:
    static constexpr uint32_t buffer_length = 100;
    using frame_type = uint32_t;
    using frame_index_type = uint32_t;
    using perf_type = float;

    enum class metric_type { MIN = 0, MAX = 1, AVERAGE = 2, MEDIAN = 3, COUNT = 4, SUM = 5 };
    static constexpr uint32_t metric_count = 6;

    MultiPerformanceHistory();

    void set_name(std::string n) {
        name = n;
    }

    const std::string& get_name() const {
        return name;
    }

    void push_sample(frame_type frame, frame_index_type idx, perf_type val);

    void reset();

    perf_type at(int index, metric_type metric) const;

    perf_type last_value(metric_type metric) const;

    perf_type average(metric_type metric) const;

    /** return statistics over the whole window/buffer in the sense of outer(inner(samples)). The outer metric summarizes the buffer, the inner metric a frame, so you can have, e.g., the average over all frame minima etc. */
    perf_type window_statistics(metric_type outer_metric, metric_type inner_metric) {
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
        case metric_type::SUM:
            return window_metrics[static_cast<uint32_t>(inner_metric)].sum();
        }
        return perf_type();
    }

    uint32_t samples() const {
        return num_samples;
    }
    uint32_t samples(int index);
    uint32_t frames() const {
        return num_frames;
    }

    /** copies the disjunct segments in the ring buffer into a contiguous array for draw calls */
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
            total += t;
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
        perf_type sum() const {
            return total;
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
            average = median = total = 0;
            minimum = std::numeric_limits<perf_type>::max();
            maximum = std::numeric_limits<perf_type>::lowest();
        }

    private:
        std::vector<perf_type> values;
        perf_type minimum = std::numeric_limits<perf_type>::max(), maximum = std::numeric_limits<perf_type>::lowest();
        perf_type average = 0, total = 0;
        mutable perf_type median = 0;
        frame_type curr_frame = std::numeric_limits<frame_type>::max();
        mutable frame_type median_computed = std::numeric_limits<frame_type>::max();
    };

    class windowed_frame_statistics {
    public:
        void push_value(perf_type t) {
            values.push_back(t);
            if (values.size() > window_size)
                values.pop_front();
            moments_ok = false;
        }
        perf_type min() {
            compute_moments();
            return minimum;
        };
        perf_type max() {
            compute_moments();
            return maximum;
        }
        perf_type avg() {
            compute_moments();
            return average;
        }
        uint32_t count() const {
            return values.size();
        }
        perf_type sum() {
            compute_moments();
            return total;
        }
        perf_type med() {
            compute_moments();
            return median;
        }
        void reset(bool clear_values = true) {
            if (clear_values)
                values.clear();
            average = median = total = 0;
            minimum = std::numeric_limits<perf_type>::max();
            maximum = std::numeric_limits<perf_type>::lowest();
        }

    private:
        void compute_moments() {
            if (!moments_ok) {
                reset(false);
                if (!values.empty()) {
                    std::vector<perf_type> copy(values.begin(), values.end());
                    std::sort(copy.begin(), copy.end());
                    median = copy[copy.size() / 2];
                    for (auto& v : copy) {
                        total += v;
                        if (v < minimum)
                            minimum = v;
                        if (v > maximum)
                            maximum = v;
                    }
                    if (copy.size() > 1)
                        average = total / copy.size();
                    else
                        average = total;
                }
                moments_ok = true;
            }
        }

        std::list<perf_type> values;
        perf_type minimum = std::numeric_limits<perf_type>::max(), maximum = std::numeric_limits<perf_type>::lowest();
        perf_type average = 0, total = 0;
        mutable perf_type median = 0;
        mutable bool moments_ok = false;
        uint32_t window_size = buffer_length;
    };

    std::array<frame_statistics, buffer_length> time_buffer{};
    std::string name;
    int next_index = 0;
    uint32_t num_samples = 0, num_frames = 0;
    std::array<windowed_frame_statistics, metric_count> window_metrics;
};

} // namespace megamol::core
