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


#include "vislib/string.h"


namespace vislib {
namespace sys {


    /**
     * Utility class for informations about the local system.
     */
    class SystemInformation {
    public:

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

