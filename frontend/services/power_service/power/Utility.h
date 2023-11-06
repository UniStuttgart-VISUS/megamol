#pragma once

#ifdef MEGAMOL_USE_POWER

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <functional>
#include <numeric>
#include <string>
#include <tuple>
#include <unordered_map>
#include <variant>
#include <vector>

#include <power_overwhelming/rtx_instrument.h>

#include "MetaData.h"
#include "Timestamp.h"

namespace megamol::power {

/// <summary>
/// Container storing a set of samples.
/// </summary>
using samples_t = std::vector<float>;
using value_map_t = std::unordered_map<std::string, std::variant<samples_t, timeline_t>>;
using segments_t = std::vector<value_map_t>;
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

//inline std::string get_name(visus::power_overwhelming::rtx_instrument const& i) {
//    auto const name_size = i.name(static_cast<char*>(nullptr), 0);
//    std::vector<char> name(name_size);
//    i.name(name.data(), name.size());
//    return std::string{name.data()};
//    /*std::string name;
//    name.resize(name_size);
//    i.name(name.data(), name.size());
//    if (!name.empty()) {
//        name.resize(name.size() - 1);
//    }
//    return name;*/
//}

//inline std::string get_identity(visus::power_overwhelming::rtx_instrument const& i) {
//    auto const id_size = i.identify((char*)nullptr, 0);
//    std::string id;
//    id.resize(id_size);
//    i.identify(id.data(), id.size());
//    if (!id.empty()) {
//        id.resize(id.size() - 1);
//    }
//    return id;
//}

/// <summary>
/// Copy a power_overwhelming waveform into vector.
/// </summary>
/// <param name="wave">The waveform.</param>
/// <returns>Vector with the samples from the waveform.</returns>
inline std::vector<float> copy_waveform(visus::power_overwhelming::oscilloscope_waveform const& wave) {
    std::vector<float> ret(wave.record_length());
    std::copy(wave.begin(), wave.end(), ret.begin());
    return ret;
}

/// <summary>
/// Generates a sequence of timestamps for the given waveform.
/// The timestamps are in filetime.
/// </summary>
/// <param name="waveform">The waveform.</param>
/// <returns>Vector containing the timestamps.</returns>
inline power::timeline_t generate_timestamps_ft(visus::power_overwhelming::oscilloscope_waveform const& waveform) {

    auto const t_begin = waveform.time_begin();
    //auto const t_end = waveform.time_end();
    auto const t_dis = waveform.sample_distance();
    //auto const t_off = waveform.segment_offset();
    auto const r_length = waveform.record_length();

    auto const t_b_ft = std::chrono::duration_cast<filetime_dur_t>(std::chrono::duration<float>(t_begin));
    auto const t_d_ft = std::chrono::duration_cast<filetime_dur_t>(std::chrono::duration<float>(t_dis));

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
/// <param name="ext">Extension of the file. (With leading '.')</param>
/// <returns>Full path to the segment file destination.</returns>
inline std::filesystem::path create_full_path(std::filesystem::path const& output_folder,
    std::string const& device_name, std::size_t const s_idx, std::string const& ext = ".parquet") {
    return output_folder / (device_name + "_s" + std::to_string(s_idx) + ext);
}

} // namespace megamol::power

#endif
