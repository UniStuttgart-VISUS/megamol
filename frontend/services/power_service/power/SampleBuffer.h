#pragma once

#include <chrono>
#include <string>
#include <vector>

#include "Utility.h"

namespace megamol::power {

class SampleBuffer {
public:
    SampleBuffer() = default;

    explicit SampleBuffer(std::string const& name, std::chrono::milliseconds const& sample_range,
        std::chrono::milliseconds const& sample_dis);

    void Add(float const sample, int64_t const timestamp);

    void Clear();

    samples_t const& ReadSamples() const {
        return samples_;
    }

    timeline_t const& ReadTimestamps() const {
        return timestamps_;
    }

    std::string const& Name() const {
        return name_;
    }

    void SetSampleRange(std::chrono::milliseconds const& sample_range);

private:
    std::string name_;

    std::size_t cap_incr_ = 100;

    samples_t samples_;

    timeline_t timestamps_;

    std::chrono::milliseconds sample_dis_;
};

} // namespace megamol::power
