/*
 * Runnable.h  15.08.2006 (mueller)
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */

#ifndef VISLIB_RUNNABLE_H_INCLUDED
#define VISLIB_RUNNABLE_H_INCLUDED
#if (_MSC_VER > 1000)
#pragma once
#endif /* (_MSC_VER > 1000) */


#include "vislib/types.h"


namespace vislib {
namespace sys {

    /**
     * Defines the interface for objects that can be run in threads.
     */
    class Runnable {

    public:

        /**
         * Perform the work of a thread.
         *
         * @param userData A pointer to user data that are passed to the thread,
         *                 if it started.
         *
         * @return The application dependent return code of the thread.
         */
        virtual DWORD Run(const void *userData) = 0;

	};

} /* end namespace sys */
} /* end namespace vislib */

#endif /* VISLIB_RUNNABLE_H_INCLUDED */
