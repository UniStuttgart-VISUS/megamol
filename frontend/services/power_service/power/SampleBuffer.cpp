#include "SampleBuffer.h"

namespace megamol::power {

SampleBuffer::SampleBuffer(
    std::string const& name, std::chrono::milliseconds const& sample_range, std::chrono::milliseconds const& sample_dis)
        : name_(name)
        , sample_dis_(sample_dis) {
    SetSampleRange(sample_range);
}

void SampleBuffer::Add(float const sample, int64_t const timestamp, int64_t const walltime) {
    samples_.push_back(sample);
    timestamps_.push_back(timestamp);
    walltimes_.push_back(walltime);
    if (samples_.size() > 0.95f * samples_.capacity()) {
        samples_.reserve(samples_.capacity() + cap_incr_);
        timestamps_.reserve(timestamps_.capacity() + cap_incr_);
        walltimes_.reserve(walltimes_.capacity() + cap_incr_);
    }
}

void SampleBuffer::Clear() {
    samples_.clear();
    timestamps_.clear();
    walltimes_.clear();
    samples_.reserve(cap_incr_);
    timestamps_.reserve(cap_incr_);
    walltimes_.reserve(cap_incr_);
}

void SampleBuffer::SetSampleRange(std::chrono::milliseconds const& sample_range) {
    auto const total_samples = sample_range / sample_dis_;
    cap_incr_ = total_samples * 1.1f;
    samples_.reserve(cap_incr_);
    timestamps_.reserve(cap_incr_);
    walltimes_.reserve(cap_incr_);
}

} // namespace megamol::power
