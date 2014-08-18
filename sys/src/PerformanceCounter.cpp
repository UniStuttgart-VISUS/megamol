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

#include "vislib/assert.h"
#include "vislib/error.h"
#include "vislib/memutils.h"
#include "vislib/SystemException.h"
#include "vislib/Trace.h"


/*
 * vislib::sys::PerformanceCounter::Query
 */
UINT64 vislib::sys::PerformanceCounter::Query(const bool useFullPrecision) {
#ifdef _WIN32
    LARGE_INTEGER timerCount;

    if (!::QueryPerformanceCounter(&timerCount)) {
        VLTRACE(Trace::LEVEL_ERROR, "QueryPerformanceCounter failed in "
            "vislib::sys::PerformanceCounter::Query\n");
        throw SystemException(__FILE__, __LINE__);
    }

    if (useFullPrecision) {
        return timerCount.QuadPart;
    } else {
        return (timerCount.QuadPart * 1000) / QueryFrequency();
    }

#else /* _WIN32 */
    struct timeval t;
    if (::gettimeofday(&t, NULL) == -1) {
        throw SystemException(__FILE__, __LINE__);
    }

    if (useFullPrecision) {
        return static_cast<UINT64>(t.tv_sec * 1e6 + t.tv_usec);
    } else {
        return static_cast<UINT64>((t.tv_sec * 1e6 + t.tv_usec) / 1000.0);
    }

#endif /* _WIN32 */
}


/*
 * vislib::sys::PerformanceCounter::QueryFrequency
 */
UINT64 vislib::sys::PerformanceCounter::QueryFrequency(void) {
#ifdef _WIN32
    LARGE_INTEGER timerFreq;

    if (!::QueryPerformanceFrequency(&timerFreq)) {
        VLTRACE(Trace::LEVEL_ERROR, "QueryPerformanceFrequency failed in "
            "vislib::sys::PerformanceCounter::Query\n");
        throw SystemException(__FILE__, __LINE__);
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
