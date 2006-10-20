/*
 * SystemInformation.cpp
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */


#include "vislib/SystemInformation.h"
#include "vislib/UnsupportedOperationException.h"
#include "vislib/SystemException.h"
#include "DynamicFunctionPointer.h"

#ifdef _WIN32
#include <windows.h>
#else
#include <sys/utsname.h>
#include <unistd.h>
#include <sys/sysinfo.h>
#include <string.h>
#include <stdio.h>
#include <errno.h>
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
 * vislib::sys::SystemInformation::GetPageSize
 */
DWORD vislib::sys::SystemInformation::GetPageSize(void) {
#ifdef _WIN32
    SYSTEM_INFO si;
    ::GetSystemInfo(&si);
    return si.dwPageSize;

#else /* _WIN32 */
    int retval = ::sysconf(_SC_PAGESIZE);

    if (retval == -1) {
        throw SystemException(__FILE__, __LINE__);
    }

    return static_cast<DWORD>(retval);

#endif /* _WIN32 */
}


/*
 * vislib::sys::SystemInformation::GetPhysicalMemorySize
 */
UINT64 vislib::sys::SystemInformation::GetPhysicalMemorySize(void) {
#ifdef _WIN32
    /*
     * It's necessary to call the ex version to get information on machines 
     * with more then 4 GB ram. However, staticly linking would restrict the
     * vislib to windowsXP pro and newer, which is a to hard restrict.
     */
    UINT64 retval = 0;
    DynamicFunctionPointer<BOOL (WINAPI*)(MEMORYSTATUSEX *)> gmsEx("kernel32", "GlobalMemoryStatusEx");

    if (gmsEx.IsValid()) {
        MEMORYSTATUSEX memStat;
        memStat.dwLength = sizeof(memStat);

        if (gmsEx(&memStat) == 0) {
            throw SystemException(__FILE__, __LINE__);
        }

        retval = memStat.ullTotalPhys;

    } else {
        MEMORYSTATUS memStat;

        GlobalMemoryStatus(&memStat);

        retval = memStat.dwTotalPhys;
    }

    return retval;
#else /* _WIN32 */
    struct sysinfo info;

    if (sysinfo(&info) != 0) {
        throw SystemException(__FILE__, __LINE__);
    }

    if (sizeof(info._f) != (sizeof(char) * (20 - 2 * sizeof(long) - sizeof(int)))) {
        /* a fucking old kernel is used */
        return static_cast<UINT64>(info.totalram);
    }
    return static_cast<UINT64>(info.totalram) 
        * static_cast<UINT64>(info.mem_unit);

#endif /* _WIN32 */
}


/*
 * vislib::sys::SystemInformation::GetAvailableMemorySize
 */
UINT64 vislib::sys::SystemInformation::GetAvailableMemorySize(void) {
#ifdef _WIN32
    /*
     * It's necessary to call the ex version to get information on machines 
     * with more then 4 GB ram. However, staticly linking would restrict the
     * vislib to windowsXP pro and newer, which is a to hard restrict.
     */
    UINT64 retval = 0;
    DynamicFunctionPointer<BOOL (WINAPI*)(MEMORYSTATUSEX *)> gmsEx("kernel32", "GlobalMemoryStatusEx");

    if (gmsEx.IsValid()) {
        MEMORYSTATUSEX memStat;
        memStat.dwLength = sizeof(memStat);

        if (gmsEx(&memStat) == 0) {
            throw SystemException(__FILE__, __LINE__);
        }

        retval = memStat.ullAvailPhys;

    } else {
        MEMORYSTATUS memStat;

        GlobalMemoryStatus(&memStat);

        retval = memStat.dwAvailPhys;
    }

    return retval;
#else /* _WIN32 */
    struct sysinfo info;

    if (sysinfo(&info) != 0) {
        throw SystemException(__FILE__, __LINE__);
    }

    if (sizeof(info._f) != (sizeof(char) * (20 - 2 * sizeof(long) - sizeof(int)))) {
        /* a fucking old kernel is used */
        return static_cast<UINT64>(info.freeram);
    }
    return static_cast<UINT64>(info.freeram) 
        * static_cast<UINT64>(info.mem_unit);

#endif /* _WIN32 */
}


/*
 * vislib::sys::SystemInformation::GetProcessorCount
 */
unsigned int vislib::sys::SystemInformation::GetProcessorCount(void) {
#ifdef _WIN32
    SYSTEM_INFO si;
    ::GetSystemInfo(&si);
    return si.dwNumberOfProcessors;

#else /* _WIN32 */
#if defined(get_nprocs)
    return static_cast<unsigned int>(get_nprocs);
#elif defined(_SC_NPROCESSORS_ONLN)
    int retval = ::sysconf(_SC_NPROCESSORS_ONLN);

    if (retval == -1) {
        throw SystemException(__FILE__, __LINE__);
    }

    return static_cast<unsigned int>(retval);
#else 
    
    // TODO: Rewrite as soon as better support classes are available
    FILE *cpuinfo = fopen("/proc/cpuinfo", "rt");
    if (cpuinfo) {
        unsigned int countCPUs = 0;
        const unsigned int lineSize = 1024;
        char line[lineSize + 1];
        line[lineSize] = 0;

        while (!feof(cpuinfo)) {
            fgets(line, lineSize, cpuinfo);

            // case sensitive check necessary
            // see /proc/cpuinfo on afro or mmoovis?? for more information
            if (strstr(line, "processor") != NULL) { 
                countCPUs++;
            }
        }

        fclose(cpuinfo);
        return countCPUs;
    }

    // errno is set by failed fopen
    throw SystemException(__FILE__, __LINE__);

    return 0; // never reached 
#endif
#endif /* _WIN32 */
}


/*
 * vislib::sys::SystemInformation::GetSystemType
 */
vislib::sys::SystemInformation::SystemType vislib::sys::SystemInformation::GetSystemType(void) {
    /* I'm currently very sure that the system type can be determined by the application type */
#ifdef _WIN32
    return SYSTEM_WINDOWS;
#else
    return SYSTEM_LINUX;
#endif
}


/*
 * vislib::sys::SystemInformation::GetSystemArchitectureType
 */
unsigned int vislib::sys::SystemInformation::GetSystemWordSize(void) {
#ifdef _WIN32
    DynamicFunctionPointer<void (WINAPI*)(SYSTEM_INFO *)> gnsi("kernel32", "GetNativeSystemInfo");
    SYSTEM_INFO info;

    if (gnsi.IsValid()) {
        gnsi(&info);
    } else {
        GetSystemInfo(&info);        
    }

    switch (info.wProcessorArchitecture) {
        case PROCESSOR_ARCHITECTURE_INTEL:
            return 32;
            break;
        case PROCESSOR_ARCHITECTURE_IA64: // no break!
        case PROCESSOR_ARCHITECTURE_AMD64:
            return 64;
            break;
        case PROCESSOR_ARCHITECTURE_UNKNOWN: // no break!
        default:
            return vislib::sys::SystemInformation::GetSelfWordSize();
            break;
    }

    assert(false); // never reached!
    return 0;
#else
    struct utsname names;

    if (uname(&names) != 0) {
        return vislib::sys::SystemInformation::GetSelfWordSize();
    }

    return (strstr(names.machine, "64") == NULL) ? 32 : 64;    
#endif
}


/*
 * vislib::sys::SystemInformation::GetSelfSystemType
 */
vislib::sys::SystemInformation::SystemType vislib::sys::SystemInformation::GetSelfSystemType(void) {
#ifdef _WIN32
    return SYSTEM_WINDOWS;
#else
    return SYSTEM_LINUX;
#endif
}


/*
 * vislib::sys::SystemInformation::GetSelfWordSize
 */
unsigned int vislib::sys::SystemInformation::GetSelfWordSize(void) {
#ifdef _WIN32
#ifdef _WIN64
    return 64;
#else
    return 32;
#endif /* _WIN64 */
#else /* _WIN 32 */
#if defined(__LP64__) || defined(_LP64)
#if (__LP64 == 1) || (_LP64 == 1)
    return 64;
#else /* (__LP64 == 1) || (_LP64 == 1) */
    return 32;
#endif /* (__LP64 == 1) || (_LP64 == 1) */
#else /* defined(__LP64__) || defined(_LP64) */
    return 32;
#endif /* defined(__LP64__) || defined(_LP64) */
#endif /* _WIN32 */
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
