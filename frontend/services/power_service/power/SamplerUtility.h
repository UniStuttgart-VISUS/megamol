#pragma once

#if MEGAMOL_USE_POWER

#include <functional>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

#include <power_overwhelming/adl_sensor.h>
#include <power_overwhelming/emi_sensor.h>
#include <power_overwhelming/msr_sensor.h>
#include <power_overwhelming/nvml_sensor.h>
#include <power_overwhelming/tinkerforge_sensor.h>

#include "SampleBuffer.h"

#ifdef MEGAMOL_USE_TRACY
#include <tracy/Tracy.hpp>
#endif

using namespace visus::power_overwhelming;

namespace megamol::power {

/// <summary>
/// Container for a set of samplers from the power-overwhelming library.
/// (key = std::string, value = T)
/// </summary>
/// <typeparam name="T">Sampler type.</typeparam>
template<typename T>
using samplers_t = std::unordered_map<std::string, T>;

/// <summary>
/// Container definitions for a set of SampleBuffers <see cref="SampleBuffer"/>.
/// </summary>
using buffers_t = std::vector<SampleBuffer>;

/// <summary>
/// Type definition for user data of the sampler functions <see cref="sample_func"/>.
/// </summary>
using context_t = std::tuple<char const*, SampleBuffer*, bool const&>;

/// <summary>
/// Signature for functions that discard power-overwhelming sensors based on their name.
/// </summary>
using discard_func_t = std::function<bool(std::string const&)>;

/// <summary>
/// Signature for functions that configure a power-overwhelming sensor.
/// </summary>
/// <typeparam name="T"></typeparam>
template<typename T>
using config_func_t = std::function<void(T&)>;

/// <summary>
/// Signature for functions that transform a power-overwhelming sensor name into a MegaMol sensor name.
/// </summary>
using transform_func_t = std::function<std::string(std::string const&)>;

/// <summary>
/// Sampling function for all power-overwhelming samplers.
/// Samples will stored only when a flag passed as user data is true <see cref="context_t"/>.
/// If tracy is active, all samples are also recorded as tracy plot.
/// </summary>
/// <param name="m">Samples from the power-overwhelimg library.</param>
/// <param name="n">Number of samples in <c>m</c>.</param>
/// <param name="usr_ptr">Passed user data that contains the name of the sensor, the buffer for the samples,
/// and a flag whether samples are currently recorded.</param>
inline void sample_func(
    wchar_t const*, visus::power_overwhelming::measurement_data const* m, std::size_t const n, void* usr_ptr) {
    auto usr_data = static_cast<context_t*>(usr_ptr);
    auto name = std::get<0>(*usr_data);
    auto buffer = std::get<1>(*usr_data);
    auto const& do_buffer = std::get<2>(*usr_data);
#ifdef MEGAMOL_USE_TRACY
    for (std::size_t i = 0; i < n; ++i) {
        TracyPlot(name, m[i].power());
    }
#endif
    if (do_buffer) {
        for (std::size_t i = 0; i < n; ++i) {
            buffer->Add(m[i].power(), m[i].timestamp());
        }
    }
}
} // namespace megamol::power

#endif
