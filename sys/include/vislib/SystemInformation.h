/*
 * SystemInformation.h
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */

#ifndef VISLIB_SYSTEMINFORMATION_H_INCLUDED
#define VISLIB_SYSTEMINFORMATION_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */

#ifndef _WIN32
#include <X11/Xlib.h>
#endif /* !_WIN32 */

#include "vislib/Array.h"
#include "vislib/Rectangle.h"
#include "vislib/String.h"
#include "vislib/types.h"


namespace vislib {
namespace sys {


    /**
     * Utility class for informations about the local system.
     */
    class SystemInformation {
    public:

        /** possible values of operating system types */
        enum OSType {
            OSTYPE_UNKNOWN,
            OSTYPE_WINDOWS,
            OSTYPE_LINUX
        };

        /** possible values of machine byte order / endianness */
        enum Endianness {
            ENDIANNESS_BIG_ENDIAN,
            ENDIANNESS_LITTLE_ENDIAN,
            ENDIANNESS_MIDDLE_ENDIAN,
            ENDIANNESS_UNKNOWN
        };

        /** This dimension defines a monitor size. */
        typedef vislib::math::Rectangle<long> MonitorRect;

        /** Array of MonitorRects. */
        typedef vislib::Array<MonitorRect> MonitorRectArray;

        /**
         * Answer the the granularity of page protection and 
         * commitment. Until we know better on Linux this is equivalent
         * to PageSize()
         *
         * @return The allocation granularity in bytes.
         *
         * @throws SystemException If the page size could not be retrieved 
         *                         (Linux only).
         */
#ifdef _WIN32
        static DWORD AllocationGranularity(void);
#else /* _WIN32 */
        static inline DWORD AllocationGranularity(void) {
            return PageSize();
        }
#endif /* _WIN32 */

        /** 
         * Return the size of available physical memory in bytes, not including
         * virtual memory from swap files.
         *
         * @return The size of available memory.
         *
         * @throws SystemException on failure.
         */
        static UINT64 AvailableMemorySize(void);

        /**
         * Returns an ansi string with the local computers name.
         *
         * @param outName The ansi string with the local computers name. The 
         *                previous content of the string might be destroied, 
         *                even if the function fails.
         *
         * @throws SystemException on failure
         */
        static void ComputerName(vislib::StringA &outName);

        /**
         * Returns an unicode string with the local computers name.
         *
         * @param outName The unicode string with the local computers name. The
         *                previous content of the string might be destroied, 
         *                even if the function fails.
         *
         * @throws SystemException on failure
         */
        static void ComputerName(vislib::StringW &outName);

        /**
         * Returns a string with the local computers name.
         *
         * @return The computer name.
         *
         * @throws SystemException on failure
         */
        inline static StringA ComputerNameA(void) {
            StringA retval;
            SystemInformation::ComputerName(retval);
            return retval;
        }

        /**
         * Returns a string with the local computers name.
         *
         * @return The computer name.
         *
         * @throws SystemException on failure
         */
        inline static StringW ComputerNameW(void) {
            StringW retval;
            SystemInformation::ComputerName(retval);
            return retval;
        }

        /**
         * Answer the size and location in the virtual desktop of all monitors 
         * attached to the system.
         *
         * @param outMonitorRects An array receiving the monitor sizes.
         *
         * @return The number of entries in 'outMonitorRects'.
         *
         * @throws SystemException If a system call required for retrieving
         *                         the information failed.
         * @throws Exception       On Linux, if the X11 display could not be
         *                         opened.
         */
        static DWORD MonitorRects(MonitorRectArray& outMonitorRects);

        /**
         * Answer the page size
         *
         * @return The page size in bytes.
         *
         * @throws SystemException If the page size could not be retrieved 
         *                         (Linux only).
         */
        static DWORD PageSize(void);

        /**
         * Return the size of physical memory in bytes.
         *
         * @return The size of physical memory.
         *
         * @throws SystemException on failure.
         */
        static UINT64 PhysicalMemorySize(void);

        /**
         * Answer the size and origin of the primary monitor. 
         *
         * @return The dimension of the primary monitor.
         *
         * @throws SystemException If a system call required for retrieving
         *                         the information failed or no monitor was
         *                         found.
         * @throws Exception       On Linux, if the X11 display could not be
         *                         opened.
         */
        static MonitorRect PrimaryMonitorRect(void);

        /**
         * Return the number of processors in the local machine.
         *
         * @return The number of processors.
         *
         * @throws SystemException on failure.
         */
        static unsigned int ProcessorCount(void);

        /**
         * Returns the type of the operating system this vislib application is
         * built for. Will usually the same value as the return value of
         * "GetSystemType".
         *
         * @return The system type of the current vislib application.
         */
        static OSType SelfSystemType(void);

        /**
         * Returns the size of a word in bit of the current vislib 
         * application. The return value is a constant value depending on the 
         * compile targets used when building the lib "vislibsys".
         *
         * @return The word size of the current vislib application.
         */
        static unsigned int SelfWordSize(void);

        /**
         * Returns the endianness of the system running this vislib application.
         *
         * @return The endianness of the machine.
         */
        inline static Endianness SystemEndianness(void) {
            UINT32 endianTestInt = 0x12345678;
            UINT8 endianTestBytes[4];
            ::memcpy(endianTestBytes, &endianTestInt, 4);
            bool machineBigEndian = ((endianTestBytes[0] == 0x12)
                && (endianTestBytes[1] == 0x34)
                && (endianTestBytes[2] == 0x56)
                && (endianTestBytes[3] == 0x78));
            bool machineMiddleEndian = ((endianTestBytes[0] == 0x34)
                && (endianTestBytes[1] == 0x12)
                && (endianTestBytes[2] == 0x78)
                && (endianTestBytes[3] == 0x56));
            bool machineLittleEndian = ((endianTestBytes[0] == 0x78)
                && (endianTestBytes[1] == 0x56)
                && (endianTestBytes[2] == 0x34)
                && (endianTestBytes[3] == 0x12));
            if (machineBigEndian) {
                return ENDIANNESS_BIG_ENDIAN;
            } else if (machineLittleEndian) {
                return ENDIANNESS_LITTLE_ENDIAN;
            } else if (machineMiddleEndian) {
                return ENDIANNESS_MIDDLE_ENDIAN;
            } else {
                return ENDIANNESS_UNKNOWN;
            }
        }

        /**
         * Returns the type of the operating system currently running this 
         * vislib application.
         *
         * @return The type of the operating system.
         */
        static OSType SystemType(void);
    
        /**
         * Answer the version of the operating system.
         *
         * @param outMajor Receives the major version.
         * @param outMinor Receives the minor version.
         *
         * @throws SystemException If the version could not be retrieved.
         */
        static void SystemVersion(DWORD& outMajor, DWORD& outMinor);

        /**
         * Returns the size of a word in bit of current operating system. This
         * value may differ from the word size of the current application,
         * since 64 bit operating systems are able to run 32 bit applications.
         *
         * @return The word size of the operating system.
         */
        static unsigned int SystemWordSize(void);

        /**
         * Returns an ansi string with the local user name running this vislib
         * application.
         *
         * @param outName The ansi string with the local user name. The 
         *                previous content of the string might be destroied, 
         *                even if the function fails.
         *
         * @throws SystemException on failure
         */
        static void UserName(vislib::StringA &outName);

        /**
         * Returns an unicode string with the local user name running this
         * vislib application
         *
         * @param outName The unicode string with the local user name. The
         *                previous content of the string might be destroied, 
         *                even if the function fails.
         *
         * @throws SystemException on failure
         */
        static void UserName(vislib::StringW &outName);

        /**
         * Returns a string with the local user name running this vislib 
         * application.
         *
         * @return The user name.
         *
         * @throws SystemException on failure
         */
        inline static StringA UserNameA(void) {
            StringA retval;
            SystemInformation::UserName(retval);
            return retval;
        }

        /**
         * Returns a string with the local user name running this vislib 
         * application.
         *
         * @return The user name.
         *
         * @throws SystemException on failure
         */
        inline static StringW UserNameW(void) {
            StringW retval;
            SystemInformation::UserName(retval);
            return retval;
        }

        /**
         * Computes the extents of the whole virtual screen formed by all
         * monitors attached to the system. 
         *
         * Please note that the screen must not fill the whole rectangle 
         * returned, but there might be holes in the screen. The returned
         * rectangle is the bounding rectangle of the virtual screen.
         *
         * @return The bounding rectangle of the virtual screen.
         */
        static MonitorRect VirtualScreen(void);

    private:

#ifdef _WIN32
        /** 
         * Callback method for computing the total virtual screen size on
         * Windows.
         *
         * @param hMonitor    Handle to display monitor.
         * @param hdcMonitor  Handle to monitor DC.
         * @param lprcMonitor Monitor intersection rectangle.
         * @param dwData      Pointer to a MonitorRect to store the dimension 
         *                    to.
         *
         * @return TRUE if the enumeration should be continued, FALSE otherwise.
         */
        static BOOL CALLBACK calcVirtualScreenProc(HMONITOR hMonitor, 
            HDC hdcMonitor, LPRECT lprcMonitor, LPARAM dwData);
#endif /* _WIN32 */

#ifndef _WIN32
        /**
         * Open the root window of the specified X11 screen and answer its 
         * location and dimension.
         *
         * @param dpy    The display.
         * @param screen The screen number.
         *
         * @return The window rectangle.
         */
        static MonitorRect getRootWndRect(Display *dpy, int screen);
#endif /* !_WIN32 */

#ifdef _WIN32
        /** 
         * Callback method for enumerating the available monitors on Windows.
         *
         * @param hMonitor    Handle to display monitor.
         * @param hdcMonitor  Handle to monitor DC.
         * @param lprcMonitor Monitor intersection rectangle.
         * @param dwData      Pointer to the MonitorRectArray to fill.
         *
         * @return TRUE if the enumeration should be continued, FALSE otherwise.
         */
        static BOOL CALLBACK monitorEnumProc(HMONITOR hMonitor, HDC hdcMonitor, 
            LPRECT lprcMonitor, LPARAM dwData);

        /** 
         * Callback method for finding the primary monitor on windows.
         *
         * @param hMonitor    Handle to display monitor.
         * @param hdcMonitor  Handle to monitor DC.
         * @param lprcMonitor Monitor intersection rectangle.
         * @param dwData      Pointer to a MonitorRect to store the dimension 
         *                    to. Nothing will be written, if the primary 
         *                    monitor could not be found.
         *
         * @return TRUE if the enumeration should be continued, FALSE otherwise.
         *
         * @throws SystemException If it was not possible to determine whether 
         *                         'hMonitor' designates the primary monitor.
         */
        static BOOL CALLBACK findPrimaryMonitorProc(HMONITOR hMonitor, 
            HDC hdcMonitor, LPRECT lprcMonitor, LPARAM dwData);
#endif /* _WIN32 */

        /** forbidden Ctor. */
        SystemInformation(void);

        /** forbidden copy Ctor. */
        SystemInformation(const SystemInformation& rhs);

        /** forbidden Dtor. */
        ~SystemInformation(void);

    };
    
} /* end namespace sys */
} /* end namespace vislib */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_SYSTEMINFORMATION_H_INCLUDED */
