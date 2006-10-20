/*
 * SystemInformation.h
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */

#ifndef VISLIB_SYSTEMINFORMATION_H_INCLUDED
#define VISLIB_SYSTEMINFORMATION_H_INCLUDED
#if (_MSC_VER > 1000)
#pragma once
#endif /* (_MSC_VER > 1000) */


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
        enum SystemType {
            SYSTEM_UNKNOWN,
            SYSTEM_WINDOWS,
            SYSTEM_LINUX
        };

        /**
         * Returns an ansi string with the local computers name.
         *
         * @param outName The ansi string with the local computers name. The 
         *                previous content of the string might be destroied, 
         *                even if the function fails.
         *
         * @throws SystemException on failure
         */
        static void GetMachineName(vislib::StringA &outName);

        /**
         * Returns an unicode string with the local computers name.
         *
         * @param outName The unicode string with the local computers name. The
         *                previous content of the string might be destroied, 
         *                even if the function fails.
         *
         * @throws SystemException on failure
         */
        static void GetMachineName(vislib::StringW &outName);

        /**
         * Answer the page size and the granularity of page protection and 
         * commitment.
         *
         * @return The page size in bytes.
         *
         * @throws SystemException If the page size could not be retrieved 
         *                         (Linux only).
         */
        static DWORD GetPageSize(void);

        /**
         * Return the size of physical memory in bytes.
         *
         * @return The size of physical memory.
         *
         * @throws SystemException on failure.
         */
        static UINT64 GetPhysicalMemorySize(void);

        /** 
         * Return the size of available physical memory in bytes, not including
         * virtual memory from swap files.
         *
         * @return The size of available memory.
         *
         * @throws SystemException on failure.
         */
        static UINT64 GetAvailableMemorySize(void);

        /**
         * Return the number of processors in the local machine.
         *
         * @return The number of processors.
         *
         * @throws SystemException on failure.
         */
        static unsigned int GetProcessorCount(void);

        /**
         * Returns the type of the operating system currently running this 
         * vislib application.
         *
         * @return The type of the operating system.
         */
        static SystemType GetSystemType(void);

        /**
         * Returns the size of a word in bit of current operating system. This
         * value may differ from the word size of the current application,
         * since 64 bit operating systems are able to run 32 bit applications.
         *
         * @return The word size of the operating system.
         */
        static unsigned int GetSystemWordSize(void);

        /**
         * Returns the type of the operating system this vislib application is
         * built for. Will usually the same value as the return value of
         * "GetSystemType".
         *
         * @return The system type of the current vislib application.
         */
        static SystemType GetSelfSystemType(void);

        /**
         * Returns the size of a word in bit of the current vislib 
         * application. The return value is a constant value depending on the 
         * compile targets used when building the lib "vislibsys".
         *
         * @return The word size of the current vislib application.
         */
        static unsigned int GetSelfWordSize(void);
    
    private:

        /** forbidden Ctor. */
        SystemInformation(void);

        /** forbidden copy Ctor. */
        SystemInformation(const SystemInformation& rhs);

        /** forbidden Dtor. */
        ~SystemInformation(void);

    };
    
} /* end namespace sys */
} /* end namespace vislib */

#endif /* VISLIB_SYSTEMINFORMATION_H_INCLUDED */

