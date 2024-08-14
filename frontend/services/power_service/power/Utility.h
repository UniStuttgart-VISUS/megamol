#pragma once

#ifdef MEGAMOL_USE_POWER

#include <chrono>
#include <filesystem>
#include <functional>
#include <locale>
#include <numeric>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_map>
#include <variant>
#include <vector>

#include <power_overwhelming/oscilloscope_waveform.h>

#include "MetaData.h"
#include "Timestamp.h"

namespace megamol::power {
/// <summary>
/// Type for measurement samples.
/// </summary>
using sample_t = float;
/// <summary>
/// Container storing a set of samples.
/// </summary>
using samples_t = std::vector<sample_t>;
/// <summary>
/// Base type for a measurement as key, value map.
/// </summary>
using value_map_t = std::unordered_map<std::string, std::variant<samples_t, timeline_t>>;
/// <summary>
/// Container storing a set of measurement segments.
/// </summary>
using segments_t = std::vector<value_map_t>;
/// <summary>
/// Function signature for writers (output_path, device_name, value_map, meta).
/// </summary>
using writer_func_t =
    std::function<void(std::filesystem::path const&, std::string const&, segments_t const&, MetaData const*)>;

/// <summary>
/// Retrieves a string, such as the name of a device, from the power_overwhelming library.
/// </summary>
/// <typeparam name="T">Class type of which the retrieve function is a member.</typeparam>
/// <param name="i">Instance to invoke the retrive function.</param>
/// <param name="func">The retrieve function.</param>
/// <returns>The retrieved string.</returns>
template<typename T>
inline std::string get_pwrowg_str(T const& i, std::size_t (T::*func)(char*, std::size_t) const) {
    auto const name_size = (i.*func)(static_cast<char*>(nullptr), 0);
    std::vector<char> name(name_size);
    (i.*func)(name.data(), name.size());
    return std::string{name.data()};
}

/// <summary>
/// Copy a power_overwhelming waveform into vector.
/// </summary>
/// <param name="wave">The waveform.</param>
/// <returns>Vector with the samples from the waveform.</returns>
inline std::vector<float> copy_waveform(visus::power_overwhelming::oscilloscope_waveform const& wave) {
    return std::vector<float>(wave.begin(), wave.end());
}

/// <summary>
/// Generates a sequence of timestamps for the given waveform.
/// The timestamps are in filetime.
/// </summary>
/// <param name="waveform">The waveform.</param>
/// <returns>Vector containing the timestamps.</returns>
inline power::timeline_t generate_timeline(visus::power_overwhelming::oscilloscope_waveform const& waveform) {

    auto const t_begin = waveform.time_begin();
    //auto const t_end = waveform.time_end();
    auto const t_dis = waveform.sample_distance();
    //auto const t_off = waveform.segment_offset();
    auto const r_length = waveform.record_length();

    using ft_float = std::chrono::duration<float, filetime_dur_t::period>;

    auto const t_b_ft =
        std::chrono::duration_cast<filetime_dur_t>(ft_float(t_begin * static_cast<double>(ft_float::period::den)));
    auto const t_d_ft =
        std::chrono::duration_cast<filetime_dur_t>(ft_float(t_dis * static_cast<double>(ft_float::period::den)));

    power::timeline_t ret(r_length, t_b_ft.count());

    auto const t_d_ft_c = t_d_ft.count();

    std::inclusive_scan(
        ret.begin(), ret.end(), ret.begin(), [&t_d_ft_c](auto const& lhs, auto const& rhs) { return lhs + t_d_ft_c; });

    return ret;
}

/// <summary>
/// Parses the log file retrieved from an HMC device.
/// </summary>
/// <param name="hmc_file">The log file as string.</param>
/// <returns>A tuple containing the meta info portion as string, the csv portion as string, and the parsed csv content as a value map.</returns>
/// <exception cref="std::runtime_error"/>
std::tuple<std::string, std::string, power::value_map_t> parse_hmc_file(std::string hmc_file);

/// <summary>Creates a full file path for a segment file.</summary>
/// <param name="output_folder">Base output folder.</param>
/// <param name="device_name">Name of the device the output data is from.
/// Is used as prefix for the filename.</param>
/// <param name="s_idx">Segment index. Is included in the filename.</param>
/// <param name="ext">Extension of the file. (Expects leading '.')</param>
/// <returns>Full path to the segment file destination.</returns>
inline std::filesystem::path create_full_path(std::filesystem::path const& output_folder,
    std::string const& device_name, std::size_t const s_idx, std::string const& ext = ".parquet") {
    return output_folder / (device_name + "_s" + std::to_string(s_idx) + ext);
}

/// <summary>
/// Creates a full file path without segment id.
/// </summary>
/// <param name="output_folder">Base output folder.</param>
/// <param name="device_name">Name of the device the output data is from.
/// Is used as prefix for the filename.</param>
/// <param name="ext">Extension of the file. (Expects leading '.')</param>
/// <returns>Full path to the file destination.</returns>
inline std::filesystem::path create_full_path(
    std::filesystem::path const& output_folder, std::string const& device_name, std::string const& ext = ".parquet") {
    return output_folder / (device_name + ext);
}

/// <summary>
/// Convert a wide char string to std::string.
/// </summary>
/// <param name="name">Wide char string.</param>
/// <returns>Char string.</returns>
inline std::string unmueller_string(wchar_t const* name) {
    // https://en.cppreference.com/w/cpp/locale/codecvt/out

    auto const& f = std::use_facet<std::codecvt<wchar_t, char, std::mbstate_t>>(std::locale());

    std::wstring internal(name);
    std::mbstate_t mb = std::mbstate_t();
    std::string external(internal.size() * f.max_length(), '\0');
    const wchar_t* from_next;
    char* to_next;

    auto const res = f.out(
        mb, &internal[0], &internal[internal.size()], from_next, &external[0], &external[external.size()], to_next);
    if (res != std::codecvt_base::ok) {
        throw std::runtime_error("could not convert string");
    }
    external.resize(to_next - &external[0]);

    return external;
}

/// <summary>
/// Computes measurement prefix, postfix, and wait time for a specified time range.
/// </summary>
/// <param name="range">Base measurement range in milliseconds.</param>
/// <returns>Tuple with prefix, postfix, and wait time in milliseconds.</returns>
inline std::tuple<std::chrono::milliseconds, std::chrono::milliseconds, std::chrono::milliseconds> get_trigger_timings(
    std::chrono::milliseconds range) {
    // config_range_ / 12, config_range_ - config_range_ / 12, std::chrono::milliseconds(1000) + config_range_
    auto const lw = std::chrono::milliseconds(200);
    auto const prefix = range / 12ll + lw;
    auto const postfix = range - (range / 12ll) + lw;
    auto const wait = range + std::chrono::milliseconds(1000);
    return std::make_tuple(prefix, postfix, wait);
}

} // namespace megamol::power

#endif
