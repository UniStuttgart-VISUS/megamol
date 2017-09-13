/*
 * AboutInfo.h
 *
 * Copyright (C) 2008 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCON_ABOUTINFO_H_INCLUDED
#define MEGAMOLCON_ABOUTINFO_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */


#include "vislib/String.h"
#include "vislib/VersionNumber.h"


namespace megamol {
namespace console {
namespace utility {

    /**
     * Static utility class wrapping all informations about the binary module.
     */
    class AboutInfo {
    public:

        /**
         * Answers a print format string for dynamic library file names 
         * following the MegaMol™ file name specifications.
         * As printf formating values a single string must be specified as
         * main name of the file. The result will be something similar like
         * this, depending on the current build and operating system:
         *  'lib<BASE>64d.so' or '<BASE>32.dll'
         */
        static const char * LibFileNameFormatString(void);

        /**
         * Prints a greeting text to 'stdout'.
         */
        static void PrintGreeting(void);

        /**
         * Logs a greeting to the vislib log object.
         */
        static void LogGreeting(void);

        /**
         * Prints a text to 'stdout' with information about the version and 
         * architecture of the console module and the loaded core.
         */
        static void PrintVersionInfo(void);

        /**
         * Log a text to the vislib log object with information about the 
         * version and architecture of the console module and the loaded core.
         */
        static void LogVersionInfo(void);

        /**
         * Answer the version number of this console application.
         *
         * @return The version number of this console application.
         */
        static vislib::VersionNumber Version(void);

        /** Logs a message of the application start time */
        static void LogStartTime(void);

    private:

#ifdef _WIN32
        /** forbidden ctor */
        AboutInfo(void) = delete;
        /** forbidden dtor */
        ~AboutInfo(void) = delete;
#else
        /** forbidden ctor */
        AboutInfo(void);
        /** forbidden dtor */
        ~AboutInfo(void);
#endif

        /**
         * Returns a string describing the version information and architecture
         * of the console application.
         *
         * @return The version information string.
         */
        static vislib::StringA consoleVersionString(void);

        /**
         * Returns the comment string of the console application.
         *
         * @return The version information comment string.
         */
        static vislib::StringA consoleCommentString(void);

        /**
         * Returns a string describing the version information and architecture
         * of the loaded core.
         *
         * @param withCopyright Flag if to include the copyright note
         *
         * @return The version information string.
         */
        static vislib::StringA coreVersionString(bool withCopyright = false);

        /**
         * Returns the comment string of the console application.
         *
         * @return The version information comment string.
         */
        static vislib::StringA coreCommentString(void);

    };

} /* end namespace utility */
} /* end namespace console */
} /* end namespace megamol */

#endif /* MEGAMOLCON_ABOUTINFO_H_INCLUDED */
