#include "SampleBuffer.h"

megamol::power::SampleBuffer::SampleBuffer(
    std::string const& name, std::chrono::milliseconds const& sample_range, std::chrono::milliseconds const& sample_dis)
        : name_(name) {
    auto const total_samples = sample_range / sample_dis;
    cap_incr_ = total_samples * 1.1f;
    samples_.reserve(cap_incr_);
    timestamps_.reserve(cap_incr_);
}

void megamol::power::SampleBuffer::Add(float const sample, int64_t const timestamp) {
    samples_.push_back(sample);
    timestamps_.push_back(timestamp);
    if (samples_.size() > 0.95f * samples_.capacity()) {
        samples_.reserve(samples_.capacity() + cap_incr_);
        timestamps_.reserve(timestamps_.capacity() + cap_incr_);
    }
}
