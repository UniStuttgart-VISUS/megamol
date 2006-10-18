/*
 * SystemInformation.cpp
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */


#include "vislib/SystemInformation.h"
#include "vislib/UnsupportedOperationException.h"
#include "vislib/SystemException.h"

#ifdef _WIN32
#include <windows.h>
#else
#include <sys/utsname.h>
#endif


/*
 * vislib::sys::SystemInformation::GetMachineName
 */
void vislib::sys::SystemInformation::GetMachineName(vislib::StringA &outName) {
#ifdef _WIN32
    unsigned long oldBufSize = MAX_COMPUTERNAME_LENGTH; // used for paranoia test
    unsigned long bufSize = MAX_COMPUTERNAME_LENGTH;
    char *buf = outName.AllocateBuffer(bufSize);

    bufSize++;
    while (!::GetComputerNameA(buf, &bufSize)) {
        unsigned int le = ::GetLastError();
        bufSize--;

        if ((le == ERROR_BUFFER_OVERFLOW) && (oldBufSize != bufSize)) {
            oldBufSize = bufSize;
            buf = outName.AllocateBuffer(bufSize);

        } else {
            throw SystemException(le, __FILE__, __LINE__);

        }
        bufSize++;
    }
#else
    struct utsname names;
    if (uname(&names) != 0) {
        throw SystemException(__FILE__, __LINE__);
    }
    outName = names.nodename;
#endif
}


/*
 * vislib::sys::SystemInformation::GetMachineName
 */
void vislib::sys::SystemInformation::GetMachineName(vislib::StringW &outName) {
#ifdef _WIN32
    unsigned long oldBufSize = MAX_COMPUTERNAME_LENGTH; // used for paranoia test
    unsigned long bufSize = MAX_COMPUTERNAME_LENGTH;
    wchar_t *buf = outName.AllocateBuffer(bufSize);

    bufSize++;
    while (!::GetComputerNameW(buf, &bufSize)) {
        unsigned int le = ::GetLastError();
        bufSize--;

        if ((le == ERROR_BUFFER_OVERFLOW) && (oldBufSize != bufSize)) {
            oldBufSize = bufSize;
            buf = outName.AllocateBuffer(bufSize);

        } else {
            throw SystemException(le, __FILE__, __LINE__);

        }
        bufSize++;
    }
#else
    vislib::StringA tmpStr;
    SystemInformation::GetMachineName(tmpStr);
    outName = tmpStr;
#endif
}


/*
 * vislib::sys::SystemInformation::SystemInformation
 */
vislib::sys::SystemInformation::SystemInformation(void) {
    throw vislib::UnsupportedOperationException("SystemInformation ctor", __FILE__, __LINE__);
}


/*
 * vislib::sys::SystemInformation::SystemInformation
 */
vislib::sys::SystemInformation::SystemInformation(const vislib::sys::SystemInformation& rhs) {
    throw vislib::UnsupportedOperationException("SystemInformation copy ctor", __FILE__, __LINE__);
}


/*
 * vislib::sys::SystemInformation::~SystemInformation
 */
vislib::sys::SystemInformation::~SystemInformation(void) {
    throw vislib::UnsupportedOperationException("SystemInformation dtor", __FILE__, __LINE__);
}
