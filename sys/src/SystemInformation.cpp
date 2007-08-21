/*
 * SystemInformation.cpp
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */


#include "vislib/SystemInformation.h"

#include "DynamicFunctionPointer.h"
#include "vislib/mathfunctions.h"
#include "vislib/SystemException.h"
#include "vislib/Trace.h"
#include "vislib/UnsupportedOperationException.h"

#ifdef _WIN32
#include <windows.h>
#include <Lmcons.h>
#else
#include <sys/utsname.h>
#include <unistd.h>
#include <sys/sysinfo.h>
#include <string.h>
#include <stdio.h>
#include <errno.h>
#include <pwd.h>
#include <X11/Xlib.h>
#endif


/*
 * vislib::sys::SystemInformation::AllocationGranularity
 */
#ifdef _WIN32
DWORD vislib::sys::SystemInformation::AllocationGranularity(void) {
    SYSTEM_INFO si;
    ::GetSystemInfo(&si);
    return si.dwAllocationGranularity;
}
#endif /* _WIN32*/


/*
 * vislib::sys::SystemInformation::AvailableMemorySize
 */
UINT64 vislib::sys::SystemInformation::AvailableMemorySize(void) {
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
 * vislib::sys::SystemInformation::ComputerName
 */
void vislib::sys::SystemInformation::ComputerName(vislib::StringA &outName) {
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
 * vislib::sys::SystemInformation::ComputerName
 */
void vislib::sys::SystemInformation::ComputerName(vislib::StringW &outName) {
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
    SystemInformation::ComputerName(tmpStr);
    outName = tmpStr;
#endif
}


///* 
// * vislib::sys::SystemInformation::DisplayDeviceCount
// */
//DWORD vislib::sys::SystemInformation::DisplayDeviceCount(void) {
//#ifdef _WIN32
//    DISPLAY_DEVICE dpyDev;
//    dpyDev.cb = sizeof(DISPLAY_DEVICE);
//    DWORD retval = 0;
//
//    while (::EnumDisplayDevices(NULL, retval, &dpyDev, 0)) {
//        retval++;
//    }
//
//    return retval; 
//
//#else /* _WIN32 */
//    return 1;
//#endif /* _WIN32 */
//}



/*
 * vislib::sys::SystemInformation::MonitorSize
 */
DWORD vislib::sys::SystemInformation::MonitorSize(
        MonitorDimArray& outMonitorSizes) {
    outMonitorSizes.Clear();

#ifdef _WIN32
    if (!::EnumDisplayMonitors(NULL, NULL, SystemInformation::monitorEnumProc,
            reinterpret_cast<LPARAM>(&outMonitorSizes))) {
        throw SystemException(__FILE__, __LINE__);
    }
#else /* _WIN32 */
    Display *dpy = NULL;
    StringA errorDesc;
    int cntScreens = 0;

    if ((dpy = ::XOpenDisplay(NULL)) == NULL) {
        errorDesc.Format("Could not open display \"%s\".", 
            ::XDisplayName(NULL));
        throw Exception(errorDesc, __FILE__, __LINE__);
    }

    cntScreens = ScreenCount(dpy);
    for (int i = 0; i < cntScreens; i++) {
        outMonitorSizes.Append(MonitorDim(DisplayWidth(dpy, i), 
            DisplayHeight(dpy, i)));
    }
    
    ::XCloseDisplay(dpy);

#endif /* _WIN32 */

    return static_cast<DWORD>(outMonitorSizes.Count());
}


/*
 * vislib::sys::SystemInformation::PageSize
 */
DWORD vislib::sys::SystemInformation::PageSize(void) {
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
 * vislib::sys::SystemInformation::PhysicalMemorySize
 */
UINT64 vislib::sys::SystemInformation::PhysicalMemorySize(void) {
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

        ::GlobalMemoryStatus(&memStat);

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
 * vislib::sys::SystemInformation::PrimaryMonitorSize
 */
vislib::sys::SystemInformation::MonitorDim 
vislib::sys::SystemInformation::PrimaryMonitorSize(void) {
#ifdef _WIN32
    RECT rect;
    ::ZeroMemory(&rect, sizeof(RECT));  // Initialise as invalid.

    if (!::EnumDisplayMonitors(NULL, NULL, 
            SystemInformation::findPrimaryMonitorProc,
            reinterpret_cast<LPARAM>(&rect))) {
        throw SystemException(__FILE__, __LINE__);
    }

    if ((rect.left == 0) && (rect.right == 0) && (rect.top == 0) 
            && (rect.bottom == 0)) {
        /* Enumeration was not successful in finding primary display. */
        throw SystemException(ERROR_NOT_FOUND, __FILE__, __LINE__);
    }

    return MonitorDim(math::Abs(rect.right - rect.left), 
        math::Abs(rect.bottom - rect.top));
#else /* _WIN32 */
    int dftScreen = 0;
    Display *dpy = NULL;
    StringA errorDesc;
    MonitorDim retval;

    if ((dpy = ::XOpenDisplay(NULL)) == NULL) {
        errorDesc.Format("Could not open display \"%s\".", 
            ::XDisplayName(NULL));
        throw Exception(errorDesc, __FILE__, __LINE__);
    }

    dftScreen = DefaultScreen(dpy);
    retval.Set(DisplayWidth(dpy, dftScreen), DisplayHeight(dpy, dftScreen));
    ::XCloseDisplay(dpy);
    return retval;
#endif /* _WIN32 */
}



/*
 * vislib::sys::SystemInformation::ProcessorCount
 */
unsigned int vislib::sys::SystemInformation::ProcessorCount(void) {
#ifdef _WIN32
    SYSTEM_INFO si;
    ::GetSystemInfo(&si);
    return si.dwNumberOfProcessors;

#else /* _WIN32 */
#if defined(_SC_NPROCESSORS_ONLN)
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
 * vislib::sys::SystemInformation::SelfSystemType
 */
vislib::sys::SystemInformation::OSType vislib::sys::SystemInformation::SelfSystemType(void) {
#ifdef _WIN32
    return OS_WINDOWS;
#else
    return OS_LINUX;
#endif
}


/*
 * vislib::sys::SystemInformation::SelfWordSize
 */
unsigned int vislib::sys::SystemInformation::SelfWordSize(void) {
#ifdef _WIN32
#ifdef _WIN64
    return 64;
#else
    return 32;
#endif /* _WIN64 */
#else /* _WIN 32 */
#if ((defined(__LP64__) || defined(_LP64) || defined(__x86_64__)) \
    && ((__LP64__ != 0) || (_LP64 != 0) || (__x86_64__ != 0)))
    return 64;
#else 
    return 32;
#endif
#endif /* _WIN32 */
}


/*
 * vislib::sys::SystemInformation::SystemType
 */
vislib::sys::SystemInformation::OSType vislib::sys::SystemInformation::SystemType(void) {
    /* I'm currently very sure that the system type can be determined by the application type */
#ifdef _WIN32
    return OS_WINDOWS;
#else
    return OS_LINUX;
#endif
}


/*
 * vislib::sys::SystemInformation::SystemVersion
 */
void vislib::sys::SystemInformation::SystemVersion(DWORD& outMajor, 
                                                   DWORD& outMinor) {
#ifdef _WIN32
    OSVERSIONINFO ver;
    ver.dwOSVersionInfoSize = sizeof(OSVERSIONINFO);
    
    if (::GetVersionEx(&ver) != TRUE) {
        throw SystemException(__FILE__, __LINE__);
    }

    outMajor = ver.dwMajorVersion;
    outMinor = ver.dwMinorVersion;

#else /* _WIN32 */
    const int BUFFER_SIZE = 512;
    char buffer[BUFFER_SIZE];
    int majorVersion = 0;
    int minorVersion = 0;
    size_t cnt = 0;
    FILE *fp = NULL;

    // TODO: Use some shell abstraction class instead of popen.
    if ((fp = ::popen("uname -r", "r")) == NULL) {
        throw SystemException(__FILE__, __LINE__);
    }

    cnt = ::fread(buffer, 1, sizeof(buffer) - 1, fp);
    ::pclose(fp);

    if (cnt == 0)  {
        throw SystemException(__FILE__, __LINE__);
    }

    if (::sscanf(buffer, "%d.%d", &majorVersion, &minorVersion) != 2) {
        TRACE(Trace::LEVEL_ERROR, "sscanf on version string failed.");
        throw SystemException(ENOTSUP, __FILE__, __LINE__);
    }

    outMajor = majorVersion;
    outMinor = minorVersion;
#endif /* _WIN32 */
}


/*
 * vislib::sys::SystemInformation::SystemWordSize
 */
unsigned int vislib::sys::SystemInformation::SystemWordSize(void) {
#ifdef _WIN32
    DynamicFunctionPointer<void (WINAPI*)(SYSTEM_INFO *)> gnsi("kernel32", "GetNativeSystemInfo");
    SYSTEM_INFO info;

    if (gnsi.IsValid()) {
        gnsi(&info);
    } else {
        ::GetSystemInfo(&info);        
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
            return vislib::sys::SystemInformation::SelfWordSize();
            break;
    }

    assert(false); // never reached!
    return 0;
#else
    struct utsname names;

    if (uname(&names) != 0) {
        return vislib::sys::SystemInformation::SelfWordSize();
    }

    return (strstr(names.machine, "64") == NULL) ? 32 : 64;    
#endif
}


/*
 * vislib::sys::SystemInformation::UserName
 */
void vislib::sys::SystemInformation::UserName(vislib::StringA &outName) {
#ifdef _WIN32
    unsigned long oldBufSize = UNLEN; // used for paranoia test
    unsigned long bufSize = UNLEN;
    char *buf = outName.AllocateBuffer(bufSize);

    bufSize++;
    while (!::GetUserNameA(buf, &bufSize)) {
        unsigned int le = ::GetLastError();
        bufSize--;

        if ((le == ERROR_INSUFFICIENT_BUFFER) && (oldBufSize != bufSize)) {
            oldBufSize = bufSize;
            buf = outName.AllocateBuffer(bufSize);

        } else {
            throw SystemException(le, __FILE__, __LINE__);

        }
        bufSize++;
    }
#else /* _WIN32 */

    /* I hate linux because it's completely impossible to write backward-compatible code */
    uid_t uid = geteuid();

    struct passwd *passwd = getpwuid(uid);
    if (passwd == NULL) {
        throw SystemException(ENOENT, __FILE__, __LINE__);
    }
    outName = passwd->pw_name;

#endif /* _WIN32 */
}


/*
 * vislib::sys::SystemInformation::UserName
 */
void vislib::sys::SystemInformation::UserName(vislib::StringW &outName) {
#ifdef _WIN32
    unsigned long oldBufSize = UNLEN; // used for paranoia test
    unsigned long bufSize = UNLEN;
    wchar_t *buf = outName.AllocateBuffer(bufSize);

    bufSize++;
    while (!::GetUserNameW(buf, &bufSize)) {
        unsigned int le = ::GetLastError();
        bufSize--;

        if ((le == ERROR_INSUFFICIENT_BUFFER) && (oldBufSize != bufSize)) {
            oldBufSize = bufSize;
            buf = outName.AllocateBuffer(bufSize);

        } else {
            throw SystemException(le, __FILE__, __LINE__);

        }
        bufSize++;
    }
#else
    vislib::StringA tmpStr;
    SystemInformation::UserName(tmpStr);
    outName = tmpStr;
#endif
}


#ifdef _WIN32
/*
 * vislib::sys::SystemInformation::monitorEnumProc
 */
BOOL CALLBACK vislib::sys::SystemInformation::monitorEnumProc(HMONITOR hMonitor,
        HDC hdcMonitor, LPRECT lprcMonitor, LPARAM dwData) {
    ASSERT(hdcMonitor == NULL);
    MonitorDimArray *da = reinterpret_cast<MonitorDimArray *>(dwData);
    
    da->Append(MonitorDim(math::Abs(lprcMonitor->right - lprcMonitor->left), 
        math::Abs(lprcMonitor->bottom - lprcMonitor->top)));

    return TRUE;
}


/*
 * vislib::sys::SystemInformation::findPrimaryMonitorProc
 */
BOOL CALLBACK vislib::sys::SystemInformation::findPrimaryMonitorProc(
        HMONITOR hMonitor, HDC hdcMonitor, LPRECT lprcMonitor, LPARAM dwData) {
    MONITORINFO mi;

    ::ZeroMemory(&mi, sizeof(MONITORINFO));
    mi.cbSize = sizeof(MONITORINFO);
    
    if (::GetMonitorInfo(hMonitor, &mi) != FALSE) {
        if ((mi.dwFlags & MONITORINFOF_PRIMARY) != 0) {
            *reinterpret_cast<LPRECT>(dwData) = *lprcMonitor;
            //return FALSE;
            // Stopping the enumeration by returning FALSE does not work at
            // least on Vista.
        }
    } else {
        throw SystemException(__FILE__, __LINE__);
    }
 
    return TRUE;
}
#endif /*_ WIN32 */


/*
 * vislib::sys::SystemInformation::SystemInformation
 */
vislib::sys::SystemInformation::SystemInformation(void) {
    throw vislib::UnsupportedOperationException("SystemInformation ctor",
        __FILE__, __LINE__);
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
