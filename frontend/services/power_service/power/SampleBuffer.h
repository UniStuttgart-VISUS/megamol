#pragma once

#ifdef MEGAMOL_USE_POWER

#include <chrono>
#include <string>
#include <vector>

#include "Timestamp.h"
#include "Utility.h"

namespace megamol::power {

/// <summary>
/// Container class for the samples obtained with the power-overwhelming library.
/// </summary>
class SampleBuffer {
public:
    /// <summary>
    /// Ctor.
    /// </summary>
    SampleBuffer() = default;

    /// <summary>
    ///
    /// </summary>
    /// <param name="name"></param>
    /// <param name="sample_range"></param>
    /// <param name="sample_dis"></param>
    explicit SampleBuffer(std::string const& name, std::chrono::milliseconds const& sample_range,
        std::chrono::milliseconds const& sample_dis);

    /// <summary>
    /// Appends a sample and corresponding timestamp to the buffer.
    /// Increases capacity if it is close to be filled.
    /// </summary>
    /// <param name="sample">Sample value.</param>
    /// <param name="timestamp">Timestamp in FILETIME.</param>
    void Add(sample_t const sample, filetime_t const timestamp);

    /// <summary>
    /// Clears the underlying buffer.
    /// </summary>
    void Clear();

    /// <summary>
    /// Get the sample buffer.
    /// </summary>
    /// <returns>Sample buffer.</returns>
    samples_t const& ReadSamples() const {
        return samples_;
    }

    /// <summary>
    /// Get the timestamp buffer.
    /// </summary>
    /// <returns>Timestamp buffer.</returns>
    timeline_t const& ReadTimestamps() const {
        return timestamps_;
    }

    /// <summary>
    /// Get the name associated with the buffer.
    /// </summary>
    /// <returns></returns>
    std::string const& Name() const {
        return name_;
    }

    /// <summary>
    /// Set the sampling time range. Influences the capacity increase wrt. the sample distance.
    /// </summary>
    /// <param name="sample_range">The time range in milliseconds.</param>
    void SetSampleRange(std::chrono::milliseconds const& sample_range);

private:
    std::string name_;

    std::size_t cap_incr_ = 100;

    samples_t samples_;

    timeline_t timestamps_;

    std::chrono::milliseconds sample_dis_;
};

} // namespace megamol::power

#endif
