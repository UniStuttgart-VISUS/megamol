/*
 * PerformanceCounter.cpp  10.08.2006 (mueller)
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 * Copyright (C) 2005 by Christoph Mueller (christoph.mueller@vis.uni-stuttgart.de). Alle Rechte vorbehalten.
 */

#include "vislib/PerformanceCounter.h"

#ifdef _WIN32
#include <windows.h>
#else /* _WIN32 */
#include <sys/time.h>
#endif /* _WIN32 */

#include "the/assert.h"
#include "vislib/error.h"
#include "the/memory.h"
#include "the/system/system_exception.h"
#include "the/trace.h"


/*
 * vislib::sys::PerformanceCounter::Query
 */
uint64_t vislib::sys::PerformanceCounter::Query(const bool useFullPrecision) {
#ifdef _WIN32
    LARGE_INTEGER timerCount;

    if (!::QueryPerformanceCounter(&timerCount)) {
        THE_TRACE(THE_TRCCHL_DEFAULT, THE_TRCLVL_ERROR, "QueryPerformanceCounter failed in "
            "vislib::sys::PerformanceCounter::Query\n");
        throw the::system::system_exception(__FILE__, __LINE__);
    }

    if (useFullPrecision) {
        return timerCount.QuadPart;
    } else {
        return (timerCount.QuadPart * 1000) / QueryFrequency();
    }

#else /* _WIN32 */
    struct timeval t;
    if (::gettimeofday(&t, NULL) == -1) {
        throw the::system::system_exception(__FILE__, __LINE__);
    }

    if (useFullPrecision) {
        return static_cast<uint64_t>(t.tv_sec * 1e6 + t.tv_usec);
    } else {
        return static_cast<uint64_t>((t.tv_sec * 1e6 + t.tv_usec) / 1000.0);
    }

#endif /* _WIN32 */
}


/*
 * vislib::sys::PerformanceCounter::QueryFrequency
 */
uint64_t vislib::sys::PerformanceCounter::QueryFrequency(void) {
#ifdef _WIN32
    LARGE_INTEGER timerFreq;

    if (!::QueryPerformanceFrequency(&timerFreq)) {
        THE_TRACE(THE_TRCCHL_DEFAULT, THE_TRCLVL_ERROR, "QueryPerformanceFrequency failed in "
            "vislib::sys::PerformanceCounter::Query\n");
        throw the::system::system_exception(__FILE__, __LINE__);
    }

    return timerFreq.QuadPart;

#else /* _WIN32 */
    return 1000 * 1000;

#endif /* _WIN32 */
}


/*
 * vislib::sys::PerformanceCounter::operator =
 */
vislib::sys::PerformanceCounter& vislib::sys::PerformanceCounter::operator =(
        const PerformanceCounter& rhs) {
    if (this != &rhs) {
        this->mark = rhs.mark;
        this->isUsingFullPrecisionMark = rhs.isUsingFullPrecisionMark;
    }

    return *this;
}
