#pragma once

#include <vector>
#include <chrono>

namespace megamol::frontend {

class SampleBuffer {
public:
    SampleBuffer() = default;

    explicit SampleBuffer(std::string const& name, std::chrono::milliseconds const& sample_range, std::chrono::milliseconds const& sample_dis);

    void Add(float const sample, int64_t const timestamp);

    std::vector<float> const& ReadSamples() const {
        return samples_;
    }

    std::vector<int64_t> const& ReadTimestamps() const {
        return timestamps_;
    }

    std::string const& Name() const {
        return name_;
    }

private:
    std::string name_;

    std::size_t cap_incr_ = 100;

    std::vector<float> samples_;

    std::vector<int64_t> timestamps_;
};

} // namespace megamol::frontend
