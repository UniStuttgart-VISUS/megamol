#pragma once

#ifdef MEGAMOL_USE_POWER

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <functional>
#include <numeric>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

#include <power_overwhelming/rtx_instrument.h>

#include "MetaData.h"

#ifdef WIN32
#include <Windows.h>
#endif

namespace megamol::power {

using timeline_t = std::vector<int64_t>;
using samples_t = std::vector<float>;
using value_map_t = std::unordered_map<std::string, std::variant<samples_t, timeline_t>>;
using segments_t = std::vector<value_map_t>;
using writer_func_t = std::function<void(std::filesystem::path const&, std::string const&, segments_t const&, MetaData const*)>;
using filetime_dur_t = std::chrono::duration<float, std::ratio<1, 10000000>>;

inline int64_t get_highres_timer() {
#ifdef WIN32
    FILETIME f;
    GetSystemTimePreciseAsFileTime(&f);
    ULARGE_INTEGER tv;
    tv.HighPart = f.dwHighDateTime;
    tv.LowPart = f.dwLowDateTime;
    return tv.QuadPart;

    /*LARGE_INTEGER t;
    QueryPerformanceCounter(&t);
    return t.QuadPart;*/
#else
#endif
}

inline std::string get_name(visus::power_overwhelming::rtx_instrument const& i) {
    auto const name_size = i.name(nullptr, 0);
    std::string name;
    name.resize(name_size);
    i.name(name.data(), name.size());
    if (!name.empty()) {
        name.resize(name.size() - 1);
    }
    return name;
}

inline std::string get_identity(visus::power_overwhelming::rtx_instrument& i) {
    auto const id_size = i.identify(nullptr, 0);
    std::string id;
    id.resize(id_size);
    i.identify(id.data(), id.size());
    if (!id.empty()) {
        id.resize(id.size() - 1);
    }
    return id;
}

inline std::chrono::nanoseconds tp_dur_to_epoch_ns(std::chrono::system_clock::time_point const& tp) {
    static auto epoch = std::chrono::system_clock::from_time_t(0);
    return std::chrono::duration_cast<std::chrono::nanoseconds>(tp - epoch);
}

inline std::vector<float> transform_waveform(visus::power_overwhelming::oscilloscope_waveform const& wave) {
    std::vector<float> ret(wave.record_length());
    std::copy(wave.begin(), wave.end(), ret.begin());
    return ret;
}

inline int64_t get_highres_timer_freq() {
#ifdef WIN32
    return 1;
    /*LARGE_INTEGER f;
    QueryPerformanceFrequency(&f);
    return f.QuadPart;*/
#else
    timespec tp;
    clock_getres(CLOCK_MONOTONIC_RAW, &tp);
    return tp.tv_nsec;
#endif
}

inline power::timeline_t generate_timestamps_ns(visus::power_overwhelming::oscilloscope_waveform const& waveform) {

    auto const t_begin = waveform.time_begin();
    //auto const t_end = waveform.time_end();
    auto const t_dis = waveform.sample_distance();
    //auto const t_off = waveform.segment_offset();
    auto const r_length = waveform.record_length();

    auto const t_b_ns = std::chrono::duration_cast<filetime_dur_t>(std::chrono::duration<float>(t_begin));
    auto const t_d_ns = std::chrono::duration_cast<filetime_dur_t>(std::chrono::duration<float>(t_dis));

    power::timeline_t ret(r_length, t_b_ns.count());

    auto const t_d_ns_c = t_d_ns.count();

    std::inclusive_scan(
        ret.begin(), ret.end(), ret.begin(), [&t_d_ns_c](auto const& lhs, auto const& rhs) { return lhs + t_d_ns_c; });

    return ret;
}

inline power::timeline_t offset_timeline(power::timeline_t const& timeline, std::chrono::nanoseconds offset) {
    power::timeline_t ret(timeline.begin(), timeline.end());

    std::transform(ret.begin(), ret.end(), ret.begin(), [o = (offset.count()/100)](auto const& val) { return val + o; });

    return ret;
}

inline int64_t get_tracy_time(int64_t base, int64_t tracy_offset) {
    /*static int64_t const frequency = get_highres_timer_freq();
    auto base_ticks =
        static_cast<int64_t>((static_cast<double>(base) / 1000. / 1000. / 1000.) * static_cast<double>(frequency));
    return base_ticks + tracy_offset;*/
    return base + tracy_offset;
}

} // namespace megamol::power

#endif
