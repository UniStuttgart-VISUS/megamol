#pragma once

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <vector>

#ifdef WIN32
#include <Windows.h>
#endif

namespace megamol::power {
using timestamp_t = int64_t;
using timeline_t = std::vector<timestamp_t>;
using filetime_dur_t = std::chrono::duration<timestamp_t, std::ratio<1, 10000000>>;

inline constexpr auto ft_offset = filetime_dur_t(116444736000000000LL);

inline filetime_dur_t get_highres_timer() {
#ifdef WIN32
    FILETIME f;
    GetSystemTimePreciseAsFileTime(&f);
    ULARGE_INTEGER tv;
    tv.HighPart = f.dwHighDateTime;
    tv.LowPart = f.dwLowDateTime;
    return filetime_dur_t(tv.QuadPart);

    /*LARGE_INTEGER t;
    QueryPerformanceCounter(&t);
    return t.QuadPart;*/
#else
#endif
}

inline filetime_dur_t tp_dur_to_epoch_ft(std::chrono::system_clock::time_point const& tp) {
    static auto epoch = std::chrono::system_clock::from_time_t(0);
    return std::chrono::duration_cast<filetime_dur_t>(tp - epoch);
}

inline timeline_t offset_timeline(timeline_t const& timeline, filetime_dur_t const& offset) {
    timeline_t ret(timeline.begin(), timeline.end());

    std::transform(ret.begin(), ret.end(), ret.begin(), [o = offset.count()](auto const& val) { return val + o; });

    return ret;
}

inline filetime_dur_t convert_tm2ft(std::tm& t) {
    return filetime_dur_t(std::mktime(&t) * 10000000LL) + ft_offset;
}
} // namespace megamol::power
