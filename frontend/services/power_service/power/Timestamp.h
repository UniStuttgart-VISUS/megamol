#pragma once

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <ratio>
#include <vector>

#ifdef WIN32
#include <Windows.h>
#endif

namespace megamol::power {
/// <summary>
/// Base type for FILETIME timestamps.
/// </summary>
using filetime_t = int64_t;
/// <summary>
/// Base container for timestamps.
/// </summary>
using timeline_t = std::vector<filetime_t>;
/// <summary>
/// FILETIME chrono duration for conversion.
/// </summary>
using filetime_dur_t = std::chrono::duration<filetime_t, std::ratio<1, 10000000>>;
/// <summary>
/// FILETIME epoch offset.
/// </summary>
inline constexpr auto ft_offset = filetime_dur_t(116444736000000000LL);

/// <summary>
/// Get current time as FILETIME.
/// </summary>
/// <returns>Timestamp as FILETIME.</returns>
inline filetime_dur_t get_highres_timer() {
#ifdef WIN32
    FILETIME f;
    GetSystemTimePreciseAsFileTime(&f);
    ULARGE_INTEGER tv;
    tv.HighPart = f.dwHighDateTime;
    tv.LowPart = f.dwLowDateTime;
    return filetime_dur_t{tv.QuadPart};

    /*LARGE_INTEGER t;
    QueryPerformanceCounter(&t);
    return t.QuadPart;*/
#else
    return filetime_dur_t{0};
#endif
}

/// <summary>
/// Adds offset to a timeline.
/// </summary>
/// <param name="timeline">Timeline to offset.</param>
/// <param name="offset">The offset in filetime.</param>
/// <returns>Returns new timeline with offset.</returns>
inline timeline_t offset_timeline(timeline_t const& timeline, filetime_dur_t const& offset) {
    timeline_t ret(timeline.size());

    std::transform(
        timeline.begin(), timeline.end(), ret.begin(), [o = offset.count()](auto const& val) { return val + o; });

    return ret;
}

/// <summary>
/// Converts <c>time.h</c> time to FILETIME.
/// </summary>
/// <param name="t">Input timestamp.</param>
/// <returns>Returns timestamp in FILETIME.</returns>
inline filetime_dur_t convert_tm2ft(std::tm& t) {
    return filetime_dur_t(std::mktime(&t) * 10000000LL) + ft_offset;
}

/// <summary>
/// Converts a <c>system_time</c> time point to FILETIME.
/// </summary>
/// <param name="tp">Input timestamp.</param>
/// <returns>Returns timestamp in FILETIME.</returns>
inline filetime_dur_t convert_systemtp2ft(std::chrono::system_clock::time_point const& tp) {
    static auto epoch = std::chrono::system_clock::from_time_t(0);
    return std::chrono::duration_cast<filetime_dur_t>(tp - epoch);
}
} // namespace megamol::power
