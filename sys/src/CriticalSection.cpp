/*
 * CriticalSection.cpp
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 * Copyright (C) 2005 by Christoph Mueller (christoph.mueller@vis.uni-stuttgart.de). Alle Rechte vorbehalten.
 */

#include "vislib/assert.h"
#include "vislib/CriticalSection.h"
#include "vislib/IllegalParamException.h"
#include "vislib/UnsupportedOperationException.h"


/*
 * vislib::sys::CriticalSection::CriticalSection
 */ 
vislib::sys::CriticalSection::CriticalSection(void) {
#ifdef _WIN32
	::InitializeCriticalSection(&this->critSect);

#else /* _WIN32 */
    // Nothing to do.

#endif /* _WIN32 */
}


/*
 * vislib::sys::CriticalSection::~CriticalSection(void) 
 */
vislib::sys::CriticalSection::~CriticalSection(void) {
#ifdef _WIN32
	::DeleteCriticalSection(&this->critSect);

#else /* _WIN32 */
	// Nothing to do.

#endif /* _WIN32 */
}


/*
 * vislib::sys::CriticalSection::Lock
 */
bool vislib::sys::CriticalSection::Lock(void) {
#ifdef _WIN32
	::EnterCriticalSection(&this->critSect);
	return true;

#else /* _WIN32 */
    return this->mutex.Lock();

#endif /* _WIN32 */
}


/*
 * vislib::sys::CrititcalSection::TryLock
 */
bool vislib::sys::CriticalSection::TryLock(void) {
#ifdef _WIN32

#if (defined(_WIN32_WINNT) && (_WIN32_WINNT >= 0x0400))
	return (::TryEnterCriticalSection(&this->critSect) != 0);
#else /* (defined(_WIN32_WINNT) && (_WIN32_WINNT >= 0x0400)) */
	ASSERT(false);
	return false;
#endif /* (defined(_WIN32_WINNT) && (_WIN32_WINNT >= 0x0400)) */

#else /* _WIN32 */
	return this->mutex.TryLock();

#endif /* _WIN32 */
}


/*
 * vislib::sys::CriticalSection::Unlock
 */
bool vislib::sys::CriticalSection::Unlock(void) {
#ifdef _WIN32
	::LeaveCriticalSection(&this->critSect);
	return true;

#else /* _WIN32 */
    return this->mutex.Unlock();

#endif /* _WIN32 */
}


/*
 * vislib::sys::CriticalSection::CriticalSection
 */
vislib::sys::CriticalSection::CriticalSection(const CriticalSection& rhs) {
    throw UnsupportedOperationException(_T("vislib::sys::CriticalSection::\
CriticalSection"), __FILE__, __LINE__);
}


/*
 * vislib::sys::CriticalSection::operator =
 */
vislib::sys::CriticalSection& vislib::sys::CriticalSection::operator =(
        const CriticalSection& rhs) {
    if (this != &rhs) {
        throw IllegalParamException(_T("rhs"), __FILE__, __LINE__);
    }

    return *this;
}
