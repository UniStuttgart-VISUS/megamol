/*
 * SystemInformation.cpp
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */


#include "vislib/SystemInformation.h"

#include "DynamicFunctionPointer.h"
#include "vislib/mathfunctions.h"
#include "the/system/system_exception.h"
#include "the/trace.h"
#include "the/not_supported_exception.h"

#include "the/not_implemented_exception.h"

#include <climits>

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
#endif


/*
 * vislib::sys::SystemInformation::AllocationGranularity
 */
#ifdef _WIN32
unsigned int vislib::sys::SystemInformation::AllocationGranularity(void) {
    SYSTEM_INFO si;
    ::GetSystemInfo(&si);
    return si.dwAllocationGranularity;
}
#endif /* _WIN32*/


/*
 * vislib::sys::SystemInformation::AvailableMemorySize
 */
uint64_t vislib::sys::SystemInformation::AvailableMemorySize(void) {
#ifdef _WIN32
    /*
     * It's necessary to call the ex version to get information on machines 
     * with more then 4 GB ram. However, staticly linking would restrict the
     * vislib to windowsXP pro and newer, which is a to hard restrict.
     */
    uint64_t retval = 0;
    DynamicFunctionPointer<BOOL (WINAPI*)(MEMORYSTATUSEX *)> gmsEx("kernel32", "GlobalMemoryStatusEx");

    if (gmsEx.IsValid()) {
        MEMORYSTATUSEX memStat;
        memStat.dwLength = sizeof(memStat);

        if (gmsEx(&memStat) == 0) {
            throw the::system::system_exception(__FILE__, __LINE__);
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
        throw the::system::system_exception(__FILE__, __LINE__);
    }

    if (sizeof(info._f) != (sizeof(char) * (20 - 2 * sizeof(long) - sizeof(int)))) {
        /* a fucking old kernel is used */
        return static_cast<uint64_t>(info.freeram);
    }
    return static_cast<uint64_t>(info.freeram) 
        * static_cast<uint64_t>(info.mem_unit);

#endif /* _WIN32 */
}


/*
 * vislib::sys::SystemInformation::ComputerName
 */
void vislib::sys::SystemInformation::ComputerName(the::astring &outName) {
#ifdef _WIN32
    unsigned long oldBufSize = MAX_COMPUTERNAME_LENGTH; // used for paranoia test
    unsigned long bufSize = MAX_COMPUTERNAME_LENGTH;
    outName = the::astring(bufSize, ' ');
    char *buf = const_cast<char *>(outName.c_str());

    bufSize++;
    while (!::GetComputerNameA(buf, &bufSize)) {
        unsigned int le = ::GetLastError();
        bufSize--;

        if ((le == ERROR_BUFFER_OVERFLOW) && (oldBufSize != bufSize)) {
            oldBufSize = bufSize;
            outName = the::astring(bufSize, ' ');
            buf = const_cast<char *>(outName.c_str());

        } else {
            throw the::system::system_exception(le, __FILE__, __LINE__);

        }
        bufSize++;
    }
#else
    struct utsname names;
    if (uname(&names) != 0) {
        throw the::system::system_exception(__FILE__, __LINE__);
    }
    outName = names.nodename;
#endif
}


/*
 * vislib::sys::SystemInformation::ComputerName
 */
void vislib::sys::SystemInformation::ComputerName(the::wstring &outName) {
#ifdef _WIN32
    unsigned long oldBufSize = MAX_COMPUTERNAME_LENGTH; // used for paranoia test
    unsigned long bufSize = MAX_COMPUTERNAME_LENGTH;
    outName = the::wstring(bufSize, ' ');
    wchar_t *buf = const_cast<wchar_t *>(outName.c_str());

    bufSize++;
    while (!::GetComputerNameW(buf, &bufSize)) {
        unsigned int le = ::GetLastError();
        bufSize--;

        if ((le == ERROR_BUFFER_OVERFLOW) && (oldBufSize != bufSize)) {
            oldBufSize = bufSize;
            outName = the::wstring(bufSize, ' ');
            buf = const_cast<wchar_t *>(outName.c_str());

        } else {
            throw the::system::system_exception(le, __FILE__, __LINE__);

        }
        bufSize++;
    }
#else
    the::astring tmpStr;
    SystemInformation::ComputerName(tmpStr);
    outName = tmpStr;
#endif
}


///* 
// * vislib::sys::SystemInformation::DisplayDeviceCount
// */
//unsigned int vislib::sys::SystemInformation::DisplayDeviceCount(void) {
//#ifdef _WIN32
//    DISPLAY_DEVICE dpyDev;
//    dpyDev.cb = sizeof(DISPLAY_DEVICE);
//    unsigned int retval = 0;
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
 * vislib::sys::SystemInformation::MonitorRects
 */
unsigned int vislib::sys::SystemInformation::MonitorRects(
        MonitorRectArray& outMonitorRects) {
    outMonitorRects.Clear();

#ifdef _WIN32
    if (!::EnumDisplayMonitors(NULL, NULL, SystemInformation::monitorEnumProc,
            reinterpret_cast<LPARAM>(&outMonitorRects))) {
        throw the::system::system_exception(__FILE__, __LINE__);
    }

#else /* _WIN32 */
    int cntScreens = 0;         // # of attached screens.
    Display *dpy = NULL;        // The display.
    the::astring errorDesc;          // For formatting an error message.
    
    if ((dpy = ::XOpenDisplay(NULL)) == NULL) {
        the::text::astring_builder::format_to(errorDesc, "Could not open display \"%s\".", 
            ::XDisplayName(NULL));
        throw the::exception(errorDesc, __FILE__, __LINE__);
    }

    cntScreens = ScreenCount(dpy);
    for (int i = 0; i < cntScreens; i++) {
        outMonitorRects.Append(SystemInformation::getRootWndRect(dpy, i));
    }
    
    ::XCloseDisplay(dpy);

#endif /* _WIN32 */

    return static_cast<unsigned int>(outMonitorRects.Count());
}


/*
 * vislib::sys::SystemInformation::PageSize
 */
unsigned int vislib::sys::SystemInformation::PageSize(void) {
#ifdef _WIN32
    SYSTEM_INFO si;
    ::GetSystemInfo(&si);
    return si.dwPageSize;

#else /* _WIN32 */
    int retval = ::sysconf(_SC_PAGESIZE);

    if (retval == -1) {
        throw the::system::system_exception(__FILE__, __LINE__);
    }

    return static_cast<unsigned int>(retval);

#endif /* _WIN32 */
}


/*
 * vislib::sys::SystemInformation::PhysicalMemorySize
 */
uint64_t vislib::sys::SystemInformation::PhysicalMemorySize(void) {
#ifdef _WIN32
    /*
     * It's necessary to call the ex version to get information on machines 
     * with more then 4 GB ram. However, staticly linking would restrict the
     * vislib to windowsXP pro and newer, which is a to hard restrict.
     */
    uint64_t retval = 0;
    DynamicFunctionPointer<BOOL (WINAPI*)(MEMORYSTATUSEX *)> gmsEx("kernel32", "GlobalMemoryStatusEx");

    if (gmsEx.IsValid()) {
        MEMORYSTATUSEX memStat;
        memStat.dwLength = sizeof(memStat);

        if (gmsEx(&memStat) == 0) {
            throw the::system::system_exception(__FILE__, __LINE__);
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
        throw the::system::system_exception(__FILE__, __LINE__);
    }

    if (sizeof(info._f) != (sizeof(char) * (20 - 2 * sizeof(long) - sizeof(int)))) {
        /* a fucking old kernel is used */
        return static_cast<uint64_t>(info.totalram);
    }
    return static_cast<uint64_t>(info.totalram) 
        * static_cast<uint64_t>(info.mem_unit);

#endif /* _WIN32 */
}


/*
 * vislib::sys::SystemInformation::PrimaryMonitorRect
 */
vislib::sys::SystemInformation::MonitorRect 
vislib::sys::SystemInformation::PrimaryMonitorRect(void) {
    MonitorRect retval;

#ifdef _WIN32
    if (!::EnumDisplayMonitors(NULL, NULL, 
            SystemInformation::findPrimaryMonitorProc,
            reinterpret_cast<LPARAM>(&retval))) {
        throw the::system::system_exception(__FILE__, __LINE__);
    }

    if (retval.IsEmpty()) {
        /* Enumeration was not successful in finding primary display. */
        throw the::system::system_exception(ERROR_NOT_FOUND, __FILE__, __LINE__);
    }

#else /* _WIN32 */
    Display *dpy = NULL;
    the::astring errorDesc;

    if ((dpy = ::XOpenDisplay(NULL)) == NULL) {
        the::text::astring_builder::format_to(errorDesc, "Could not open display \"%s\".", 
            ::XDisplayName(NULL));
        throw the::exception(errorDesc, __FILE__, __LINE__);
    }

    retval = SystemInformation::getRootWndRect(dpy, DefaultScreen(dpy));
    ::XCloseDisplay(dpy);
#endif /* _WIN32 */

    return retval;
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
        throw the::system::system_exception(__FILE__, __LINE__);
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
    throw the::system::system_exception(__FILE__, __LINE__);

    return 0; // never reached 
#endif
#endif /* _WIN32 */
}


/*
 * vislib::sys::SystemInformation::SelfSystemType
 */
vislib::sys::SystemInformation::OSType vislib::sys::SystemInformation::SelfSystemType(void) {
#ifdef _WIN32
    return OSTYPE_WINDOWS;
#else
    return OSTYPE_LINUX;
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
    return OSTYPE_WINDOWS;
#else
    return OSTYPE_LINUX;
#endif
}


/*
 * vislib::sys::SystemInformation::SystemVersion
 */
void vislib::sys::SystemInformation::SystemVersion(unsigned int& outMajor, 
                                                   unsigned int& outMinor) {
#ifdef _WIN32
    OSVERSIONINFO ver;
    ver.dwOSVersionInfoSize = sizeof(OSVERSIONINFO);
    
    if (::GetVersionEx(&ver) != TRUE) {
        throw the::system::system_exception(__FILE__, __LINE__);
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
        throw the::system::system_exception(__FILE__, __LINE__);
    }

    cnt = ::fread(buffer, 1, sizeof(buffer) - 1, fp);
    ::pclose(fp);

    if (cnt == 0)  {
        throw the::system::system_exception(__FILE__, __LINE__);
    }

    if (::sscanf(buffer, "%d.%d", &majorVersion, &minorVersion) != 2) {
        THE_TRACE(THE_TRCCHL_DEFAULT, THE_TRCLVL_ERROR, "sscanf on version string failed.");
        throw the::system::system_exception(ENOTSUP, __FILE__, __LINE__);
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
void vislib::sys::SystemInformation::UserName(the::astring &outName) {
#ifdef _WIN32
    unsigned long oldBufSize = UNLEN; // used for paranoia test
    unsigned long bufSize = UNLEN;
    outName = the::astring(bufSize, ' ');
    char *buf = const_cast<char *>(outName.c_str());

    bufSize++;
    while (!::GetUserNameA(buf, &bufSize)) {
        unsigned int le = ::GetLastError();
        bufSize--;

        if ((le == ERROR_INSUFFICIENT_BUFFER) && (oldBufSize != bufSize)) {
            oldBufSize = bufSize;
            outName = the::astring(bufSize, ' ');
            buf = const_cast<char *>(outName.c_str());

        } else {
            throw the::system::system_exception(le, __FILE__, __LINE__);

        }
        bufSize++;
    }
#else /* _WIN32 */

    /* I hate linux because it's completely impossible to write backward-compatible code */
    uid_t uid = geteuid();

    struct passwd *passwd = getpwuid(uid);
    if (passwd == NULL) {
        throw the::system::system_exception(ENOENT, __FILE__, __LINE__);
    }
    outName = passwd->pw_name;

#endif /* _WIN32 */
}


/*
 * vislib::sys::SystemInformation::UserName
 */
void vislib::sys::SystemInformation::UserName(the::wstring &outName) {
#ifdef _WIN32
    unsigned long oldBufSize = UNLEN; // used for paranoia test
    unsigned long bufSize = UNLEN;
    outName = the::wstring(bufSize, ' ');
    wchar_t *buf = const_cast<wchar_t *>(outName.c_str());

    bufSize++;
    while (!::GetUserNameW(buf, &bufSize)) {
        unsigned int le = ::GetLastError();
        bufSize--;

        if ((le == ERROR_INSUFFICIENT_BUFFER) && (oldBufSize != bufSize)) {
            oldBufSize = bufSize;
            outName = the::wstring(bufSize, ' ');
            buf = const_cast<wchar_t *>(outName.c_str());

        } else {
            throw the::system::system_exception(le, __FILE__, __LINE__);

        }
        bufSize++;
    }
#else
    the::astring tmpStr;
    SystemInformation::UserName(tmpStr);
    outName = tmpStr;
#endif
}


/*
 * vislib::sys::SystemInformation::VirtualScreen
 */
vislib::sys::SystemInformation::MonitorRect 
vislib::sys::SystemInformation::VirtualScreen(void) {
    MonitorRect retval(LONG_MAX, LONG_MAX, LONG_MIN, LONG_MIN);

#ifdef _WIN32
    if (!::EnumDisplayMonitors(NULL, NULL, 
            SystemInformation::calcVirtualScreenProc, 
            reinterpret_cast<LPARAM>(&retval))) {
        throw the::system::system_exception(__FILE__, __LINE__);
    }

#else /* _WIN32 */
    MonitorRectArray monitors;
    SystemInformation::MonitorRects(monitors);

    for (size_t i = 0; i < monitors.Count(); i++) {
        const MonitorRect& monitor = monitors[i];
        if (monitor.Left() < retval.Left()) {
            retval.SetLeft(monitor.Left());
        }
        if (monitor.Bottom() < retval.Bottom()) {
            retval.SetBottom(monitor.Bottom());
        }
        if (monitor.Right() > retval.Right()) {
            retval.SetRight(monitor.Right());
        }
        if (monitor.Top() > retval.Top()) {
            retval.SetTop(monitor.Top());
        }
    }
#endif /* _WIN32 */

    return retval;
}


#ifdef _WIN32
/*
 * vislib::sys::SystemInformation::calcVirtualScreenProc
 */
BOOL CALLBACK vislib::sys::SystemInformation::calcVirtualScreenProc(
        HMONITOR hMonitor, HDC hdcMonitor, LPRECT lprcMonitor, LPARAM dwData) {
    MonitorRect *vs = reinterpret_cast<MonitorRect *>(dwData);    

    if (lprcMonitor->left < vs->Left()) {
        vs->SetLeft(lprcMonitor->left);
    }
    if (lprcMonitor->bottom < vs->Bottom()) {
        vs->SetBottom(lprcMonitor->bottom);
    }
    if (lprcMonitor->right > vs->Right()) {
        vs->SetRight(lprcMonitor->right);
    }
    if (lprcMonitor->top > vs->Top()) {
        vs->SetTop(lprcMonitor->top);
    }

    return TRUE;
}
#endif /* _WIN32 */


#ifndef _WIN32
/*
 * vislib:sys::SystemInformation::getRootWndRect
 */
vislib::sys::SystemInformation::MonitorRect 
vislib::sys::SystemInformation::getRootWndRect(Display *dpy, int screen) {
    XWindowAttributes attribs;  // Attributes of screen root window.
    Window wnd;                 // The per-screen root window.
    
    wnd = RootWindow(dpy, screen);
    ::XGetWindowAttributes(dpy, wnd, &attribs);
    // TODO: Error handling.

    return MonitorRect(attribs.x, attribs.y, attribs.width, attribs.height);
}
#endif /* !_WIN32 */


#ifdef _WIN32
/*
 * vislib::sys::SystemInformation::monitorEnumProc
 */
BOOL CALLBACK vislib::sys::SystemInformation::monitorEnumProc(HMONITOR hMonitor,
        HDC hdcMonitor, LPRECT lprcMonitor, LPARAM dwData) {
    THE_ASSERT(hdcMonitor == NULL);
    MonitorRectArray *da = reinterpret_cast<MonitorRectArray *>(dwData);
    
    da->Append(MonitorRect(lprcMonitor->left, lprcMonitor->bottom,
        lprcMonitor->right, lprcMonitor->top));

    return TRUE;
}


/*
 * vislib::sys::SystemInformation::findPrimaryMonitorProc
 */
BOOL CALLBACK vislib::sys::SystemInformation::findPrimaryMonitorProc(
        HMONITOR hMonitor, HDC hdcMonitor, LPRECT lprcMonitor, LPARAM dwData) {
    MONITORINFO mi;
    MonitorRect *ma = reinterpret_cast<MonitorRect *>(dwData);

    ::ZeroMemory(&mi, sizeof(MONITORINFO));
    mi.cbSize = sizeof(MONITORINFO);
    
    if (::GetMonitorInfo(hMonitor, &mi) != FALSE) {
        if ((mi.dwFlags & MONITORINFOF_PRIMARY) != 0) {
            ma->Set(lprcMonitor->left, lprcMonitor->bottom, lprcMonitor->right, 
                lprcMonitor->top);
            //return FALSE;
            // Stopping the enumeration by returning FALSE does not work at
            // least on Vista.
        }
    } else {
        throw the::system::system_exception(__FILE__, __LINE__);
    }
 
    return TRUE;
}
#endif /*_ WIN32 */


/*
 * vislib::sys::SystemInformation::SystemInformation
 */
vislib::sys::SystemInformation::SystemInformation(void) {
    throw the::not_supported_exception("SystemInformation ctor",
        __FILE__, __LINE__);
}


/*
 * vislib::sys::SystemInformation::SystemInformation
 */
vislib::sys::SystemInformation::SystemInformation(
        const vislib::sys::SystemInformation& rhs) {
    throw the::not_supported_exception("SystemInformation copy ctor", 
        __FILE__, __LINE__);
}


/*
 * vislib::sys::SystemInformation::~SystemInformation
 */
vislib::sys::SystemInformation::~SystemInformation(void) {
    throw the::not_supported_exception("SystemInformation dtor", 
        __FILE__, __LINE__);
}
