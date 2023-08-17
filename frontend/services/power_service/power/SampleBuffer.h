#pragma once

#include <chrono>
#include <vector>

namespace megamol::power {

class SampleBuffer {
public:
    SampleBuffer() = default;

    explicit SampleBuffer(std::string const& name, std::chrono::milliseconds const& sample_range,
        std::chrono::milliseconds const& sample_dis);

    void Add(float const sample, int64_t const timestamp, int64_t const walltime);

    void Clear();

    std::vector<float> const& ReadSamples() const {
        return samples_;
    }

    std::vector<int64_t> const& ReadTimestamps() const {
        return timestamps_;
    }

    std::vector<int64_t> const& ReadWalltimes() const {
        return walltimes_;
    }

    std::string const& Name() const {
        return name_;
    }

    void SetSampleRange(std::chrono::milliseconds const& sample_range);

private:
    std::string name_;

    std::size_t cap_incr_ = 100;

    std::vector<float> samples_;

    std::vector<int64_t> timestamps_;

    std::vector<int64_t> walltimes_;

    std::chrono::milliseconds sample_dis_;
};

} // namespace megamol::power
